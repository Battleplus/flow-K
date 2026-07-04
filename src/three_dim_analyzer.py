"""
三维综合分析入口 — FLOW AI Trader

串联消息面(第一层) + 板块ETF轮动(第二层) + 个股K线(第三层)
实现: 新闻 → 命中板块 → 确认资金 → 筛选个股 → 三维评分排序

用法:
    python src/three_dim_analyzer.py --news "NVIDIA发布新AI芯片..."
    python src/three_dim_analyzer.py --ticker NVDA
    python src/three_dim_analyzer.py --sector-snapshot
"""

import os
import sys
import argparse
import pandas as pd

# 确保从项目根目录运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.news_analysis import analyze_news
from src.sector_rotation import get_sector_snapshot, SECTOR_ETF_MAP, ALL_SECTORS
from src.screener import analyze_ticker

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
COMPANY_INFO = os.path.join(DATA_DIR, 'company_info.csv')

# 板块名称映射：company_info.csv → 标准化板块名
SECTOR_NAME_MAP = {
    'Information Technology': 'Technology',
    'Health Care': 'Healthcare',
    'Technology': 'Technology',
    'Healthcare': 'Healthcare',
    'Financials': 'Financials',
    'Energy': 'Energy',
    'Industrials': 'Industrials',
    'Consumer Discretionary': 'Consumer Discretionary',
    'Consumer Staples': 'Consumer Staples',
    'Utilities': 'Utilities',
    'Materials': 'Materials',
    'Real Estate': 'Real Estate',
    'Communication Services': 'Communication Services',
}


def _normalize_sector(raw_sector: str) -> str:
    """标准化板块名称"""
    return SECTOR_NAME_MAP.get(raw_sector, raw_sector)


def get_sector_tickers(sector: str) -> list:
    """从 company_info.csv 获取某板块的所有成分股（兼容不同命名）"""
    df = pd.read_csv(COMPANY_INFO)
    df['_norm_sector'] = df['Sector'].apply(_normalize_sector)
    tickers = df[df['_norm_sector'] == sector]['Symbol'].tolist()
    return tickers


def stock_score_to_3d(consensus_direction: str, aggregate_score: float) -> float:
    """将个股策略评分映射到三维体系的 -2~+2"""
    # screener 的 direction: strong_bullish/bullish/neutral/bearish/strong_bearish
    # aggregate_score 已经是 -N ~ +N
    direction_map = {
        'strong_bullish': 2.0,
        'bullish': 1.0,
        'neutral': 0.0,
        'bearish': -1.0,
        'strong_bearish': -2.0,
    }
    base = direction_map.get(consensus_direction, 0.0)
    # aggregate_score 微调（±0.5范围）
    adj = max(-0.5, min(0.5, aggregate_score * 0.2))
    return round(max(-2.0, min(2.0, base + adj)), 1)


def three_dim_scan(news_text: str = None, period: str = '6mo', top_n: int = 10) -> dict:
    """
    三维全链路扫描：消息面 → 板块ETF → 个股K线 → 三维评分排序

    Args:
        news_text: 新闻文本（可选，不传则只用板块轮动+个股）
        period: K线数据窗口
        top_n: 返回TOP-N股票

    Returns:
        {
            news_analysis: {...},      # 消息面分析结果
            sector_snapshot: {...},     # 板块快照
            target_sectors: [...],      # 目标板块（消息利好+资金流入交叉）
            stocks: [...],              # 推荐股票列表(按三维评分排序)
            summary: str,               # 一句话总结
        }
    """
    result = {}

    # ======== 第一层：消息面 ========
    if news_text and news_text.strip():
        print("[三维分析] 第一层：消息面分析...")
        news_result = analyze_news(news_text)
        result['news_analysis'] = news_result
        # 消息面利好的板块（score >= 1.0）
        news_bullish = [s for s, v in news_result['sector_scores'].items() if v >= 1.0]
        # 消息面利空的板块（score <= -1.0）
        news_bearish = [s for s, v in news_result['sector_scores'].items() if v <= -1.0]
    else:
        news_result = None
        news_bullish = []
        news_bearish = []

    # ======== 第二层：板块ETF轮动 ========
    print("[三维分析] 第二层：板块ETF资金轮动...")
    sector_snap = get_sector_snapshot()
    result['sector_snapshot'] = sector_snap

    # 资金流入板块（flow_score >= 1）
    flow_bullish = [s['sector'] for s in sector_snap['sectors'] if s['flow_score'] >= 1]
    # 资金流出板块
    flow_bearish = [s['sector'] for s in sector_snap['sectors'] if s['flow_score'] <= -1]

    # ======== 确定目标板块 ========
    if news_text:
        # 有新闻：消息利好 ∩ 资金流入 = 目标板块（双因子共振）
        target_sectors = [s for s in news_bullish if s in flow_bullish]
        if not target_sectors:
            # 退一步：至少消息利好或资金流入
            target_sectors = list(set(news_bullish + flow_bullish))
        if not target_sectors:
            # 再退一步：取RS最强TOP3
            target_sectors = [s['sector'] for s in sector_snap['sectors'][:3]]

        # 一票否决：消息面强利空(-2) + 资金流出 → 绝对回避
        veto_sectors = [s for s in news_bearish if s in flow_bearish]
    else:
        # 无新闻：只用资金流入板块
        target_sectors = flow_bullish if flow_bullish else [s['sector'] for s in sector_snap['sectors'][:3]]
        veto_sectors = []

    result['target_sectors'] = target_sectors
    result['veto_sectors'] = veto_sectors

    # ======== 第三层：个股K线扫描 ========
    print(f"[三维分析] 第三层：个股K线扫描（目标板块: {target_sectors}）...")

    all_stocks = []
    for sector in target_sectors:
        tickers = get_sector_tickers(sector)
        print(f"  {sector}: {len(tickers)}只成分股")

        # 板块层面的评分（来自消息面+资金轮动）
        news_score = news_result['sector_scores'].get(sector, 0.0) if news_result else 0.0
        flow_score = 0.0
        for s in sector_snap['sectors']:
            if s['sector'] == sector:
                flow_score = float(s['flow_score'])
                break
        sector_score = news_score * 0.4 + flow_score * 0.6

        for ticker in tickers:
            try:
                stock_info = analyze_ticker(ticker, period=period)
                if stock_info is None:
                    continue
            except Exception:
                continue

            # 一票否决
            if sector in veto_sectors:
                continue

            stock_3d = stock_score_to_3d(
                stock_info['consensus_direction'],
                stock_info['aggregate_score']
            )

            # 三维综合评分
            final_score = news_score * 0.25 + sector_score * 0.35 + stock_3d * 0.4
            final_score = round(max(-3.0, min(3.0, final_score)), 1)

            all_stocks.append({
                'ticker': ticker,
                'sector': sector,
                'news_score': round(news_score, 1),
                'sector_score': round(sector_score, 1),
                'stock_score': stock_3d,
                'final_score': final_score,
                'consensus': stock_info['consensus_text'],
                'consensus_direction': stock_info['consensus_direction'],
                'aggregate_score': stock_info['aggregate_score'],
                'avg_sharpe_top5': stock_info['avg_sharpe_top5'],
                'pf_return': stock_info['pf_return'],
                'close': stock_info['close'],
                'market_regime': stock_info['market_regime'],
                'signal_net': stock_info['signal_net'],
            })

    # 按三维综合评分排序
    all_stocks.sort(key=lambda x: x['final_score'], reverse=True)

    # TOP-N
    result['stocks'] = all_stocks[:top_n]
    result['total_scanned'] = len(all_stocks)

    # ======== 生成总结 ========
    top3 = result['stocks'][:3]
    top_names = ', '.join(s['ticker'] for s in top3)
    avg_score = round(sum(s['final_score'] for s in result['stocks'][:top_n]) / max(len(result['stocks'][:top_n]), 1), 1)

    if avg_score >= 1.5:
        action = '强烈看多'
    elif avg_score >= 0.5:
        action = '偏多'
    elif avg_score >= -0.5:
        action = '观望'
    elif avg_score >= -1.5:
        action = '偏空'
    else:
        action = '强烈看空'

    result['summary'] = (
        f"三维综合评分: {action} (均值{avg_score}) | "
        f"TOP3: {top_names} | "
        f"目标板块: {', '.join(target_sectors[:3])}"
    )

    if veto_sectors:
        result['summary'] += f" | 回避板块: {', '.join(veto_sectors[:3])}"

    return result


def three_dim_single(ticker: str, period: str = '6mo') -> dict:
    """单只股票的三维评分详情"""
    # 确定板块
    df = pd.read_csv(COMPANY_INFO)
    row = df[df['Symbol'] == ticker.upper()]
    if row.empty:
        return {'error': f'股票 {ticker} 不在 company_info 中'}
    sector_raw = row.iloc[0]['Sector']
    sector = _normalize_sector(sector_raw)

    # 板块快照
    snap = get_sector_snapshot()
    sector_info = None
    for s in snap['sectors']:
        if s['sector'] == sector:
            sector_info = s
            break

    # 个股分析
    stock_info = analyze_ticker(ticker, period=period)
    if stock_info is None:
        return {'error': f'无法获取 {ticker} 的数据'}

    stock_3d = stock_score_to_3d(stock_info['consensus_direction'], stock_info['aggregate_score'])

    flow_score = sector_info['flow_score'] if sector_info else 0
    sector_score = float(flow_score)  # 没有消息面时，板块分=资金流向分
    final_score = sector_score * 0.35 + stock_3d * 0.4 + 0 * 0.25  # 消息面=0

    return {
        'ticker': ticker,
        'sector': sector,
        'news_score': 0.0,
        'sector_score': round(sector_score, 1),
        'stock_score': stock_3d,
        'final_score': round(final_score, 1),
        'stock_info': stock_info,
        'sector_info': sector_info,
    }


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FLOW 三维综合分析')
    parser.add_argument('--news', '-n', type=str, help='新闻文本')
    parser.add_argument('--ticker', '-t', type=str, help='单只股票分析')
    parser.add_argument('--sector-snapshot', '-s', action='store_true', help='仅显示板块快照')
    parser.add_argument('--period', '-p', type=str, default='6mo', help='K线周期(1mo/3mo/6mo/1y)')
    parser.add_argument('--top', type=int, default=10, help='返回TOP-N股票')
    args = parser.parse_args()

    if args.sector_snapshot:
        snap = get_sector_snapshot()
        print(f"\n市场状态: {snap['market_state']}")
        print(f"强势板块({snap['bullish_count']}): {[s['sector'] for s in snap['top_sectors']]}")
        print(f"弱势板块({snap['bearish_count']}): {[s['sector'] for s in snap['weak_sectors']]}")
        if snap['rotation_signals']:
            print(f"\n轮动信号:")
            for sig in snap['rotation_signals']:
                print(f"  [{sig['type']}] {sig['message']}")
    elif args.ticker:
        result = three_dim_single(args.ticker, period=args.period)
        if 'error' in result:
            print(f"错误: {result['error']}")
        else:
            print(f"\n{'='*60}")
            print(f"🔍 {result['ticker']} 三维分析")
            print(f"{'='*60}")
            print(f"板块: {result['sector']}")
            print(f"消息面评分: {result['news_score']:.1f}")
            print(f"板块评分: {result['sector_score']:.1f}")
            print(f"个股评分: {result['stock_score']:.1f}")
            print(f"三维综合: {result['final_score']:.1f}")
            print(f"\n个股详情: {result['stock_info']['consensus_text']}, "
                  f"夏普(Top5)={result['stock_info']['avg_sharpe_top5']}, "
                  f"组合收益={result['stock_info']['pf_return']}%")
    else:
        news = args.news or ""
        print(f"\n{'='*60}")
        print(f"🚀 FLOW 三维分析")
        print(f"{'='*60}")
        if news:
            print(f"📰 新闻: {news[:100]}...")

        result = three_dim_scan(news_text=news, period=args.period, top_n=args.top)

        print(f"\n📊 {result['summary']}")
        print(f"\n{'排名':4s} {'代码':6s} {'板块':16s} {'消息面':5s} {'板块分':5s} {'个股分':5s} {'三维综合':7s} {'共识':8s} {'夏普':6s}")
        print("-" * 75)
        for i, s in enumerate(result['stocks']):
            print(f"{i+1:3d}. {s['ticker']:6s} {s['sector']:16s} "
                  f"{s['news_score']:5.1f} {s['sector_score']:5.1f} {s['stock_score']:5.1f} "
                  f"{s['final_score']:6.1f}   {s['consensus']:8s} {s['avg_sharpe_top5']:6.2f}")

        if result.get('veto_sectors'):
            print(f"\n⛔ 一票否决板块: {result['veto_sectors']}")
