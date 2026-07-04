"""
板块ETF资金轮动模块 — FLOW AI Trader 三维框架第二层

核心逻辑：资金总量守恒，板块相对强度就是资金流向的"水表"
    不需要 Level2 资金流数据，看11个板块ETF的相对排名就知道钱在往哪流

功能：
- etf_trend_score(): 对单只ETF跑K线技术分析，复用现有 indicators/patterns/strategies
- relative_strength(): 计算超额收益(ETF - SPY)，按排名打分
- flow_matrix(): 趋势 × 相对强度 交叉判断资金流向
- get_sector_snapshot(): 11个板块完整快照（所有评分+排名+轮动信号）
"""

import os
import sys
import numpy as np
import pandas as pd

# 添加 src 目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import fetch_data
from patterns import add_indicators, detect_all
from strategies import get_latest_consensus
from indicators import ma_alignment

# ============================================================
# 板块定义
# ============================================================
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Utilities': 'XLU',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
}

ALL_SECTORS = list(SECTOR_ETF_MAP.keys())
ALL_ETFS = list(SECTOR_ETF_MAP.values())

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
ETF_CSV = os.path.join(DATA_DIR, 'sector_etf_prices.csv')


def _load_etf_from_csv(ticker: str) -> pd.DataFrame:
    """从本地 sector_etf_prices.csv 加载ETF数据"""
    df = pd.read_csv(ETF_CSV, parse_dates=['Date'])
    # ETF CSV 用 Ticker 列
    sub = df[df['Ticker'].str.upper() == ticker.upper()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.sort_values('Date').reset_index(drop=True)
    sub['Ticker'] = ticker.upper()
    return sub


def etf_trend_score(ticker: str, period: str = '6mo') -> dict:
    """
    对ETF跑完整技术分析，汇聚出趋势评分

    复用链路: fetch_data → add_indicators → detect_all → get_latest_consensus → ma_alignment

    Returns:
        {
            score: int,          # -1(空头) / 0(中性/交织) / 1(多头)
            label: str,          # "多头排列" / "空头排列" / "均线交织" / "趋势不明"
            ma_detail: dict,     # {ma_shift_3d, ma_slope_20d, ma_alignment}
            consensus: dict,     # 多策略共识
            signals_active: int, # 活跃买入信号数
            signals: list,       # 活跃信号列表
            price: float,        # 最新收盘价
            change_pct: float,   # 近5日涨跌幅
        }
    """
    # 1. 加载数据
    df = _load_etf_from_csv(ticker)
    if df.empty:
        return _empty_trend_result()

    # 2. 在 period 窗口内截取
    if period:
        # period格式: "6mo" → 180天, "1y" → 365天
        period_days = {'6mo': 180, '1y': 365, '2y': 730}.get(period, 180)
        cutoff = pd.Timestamp.now() - pd.DateOffset(days=period_days)
        df = df[df['Date'] >= cutoff].copy()
        if len(df) < 20:
            return _empty_trend_result()

    # 3. 增加指标
    df = add_indicators(df)

    # 4. 检测信号
    df = detect_all(df)

    # 5. 获取策略共识
    try:
        consensus = get_latest_consensus(df)
    except Exception:
        consensus = {'consensus': 'neutral', 'aggregate_score': 0.0, 'bullish_ratio': 0.0}

    # 6. 均线排列（用于判断中期趋势）
    try:
        ma_info = ma_alignment(df)
    except Exception:
        ma_info = {'alignment': '交织', 'sorted_mas': []}

    # 7. 汇总趋势评分
    latest = df.iloc[-1]
    price = float(latest.get('Close', latest.get('_price', 0)))
    prev_5d = df.iloc[-6]['Close'] if len(df) > 6 else df.iloc[0]['Close']
    change_pct = round((price - prev_5d) / prev_5d * 100, 2)

    # 策略层面的方向
    cons_dir = consensus.get('consensus', 'neutral')
    agg_score = float(consensus.get('aggregate_score', 0.0))

    # 均线层面
    ma_align = ma_info.get('alignment', '交织')

    # 综合判断
    points = 0
    if ma_align == '多头排列':
        points += 2
    elif ma_align == '空头排列':
        points -= 2
    else:
        pass  # 交织不加分

    if cons_dir == 'bullish':
        points += 1
    elif cons_dir == 'bearish':
        points -= 1

    # 价格位置：高于MA20加1分，低于减1分
    ma20 = latest.get('ma20', None)
    if ma20 and ma20 > 0:
        if price > ma20:
            points += 1
        else:
            points -= 1

    if points >= 2:
        score, label = 1, '多头趋势'
    elif points <= -2:
        score, label = -1, '空头趋势'
    else:
        score, label = 0, '方向不明'

    # 活跃信号
    signal_cols = [c for c in df.columns if c.startswith('signal_')]
    active_signals = []
    if signal_cols:
        recent = df.iloc[-5:]  # 最近5天
        for col in signal_cols:
            if recent[col].sum() > 0:
                name = col.replace('signal_', '').replace('_', ' ')
                active_signals.append(name)

    return {
        'score': score,
        'label': label,
        'ma_detail': {
            'alignment': ma_align,
            'sorted_mas': ma_info.get('sorted_mas', []),
        },
        'consensus': {
            'direction': cons_dir,
            'aggregate_score': round(agg_score, 2),
            'bullish_ratio': round(float(consensus.get('bullish_ratio', 0.0)), 2),
        },
        'signals_active': len(active_signals),
        'signals': active_signals[:5],
        'price': round(price, 2),
        'change_5d': change_pct,
    }


def _empty_trend_result():
    return {
        'score': 0, 'label': '数据不足',
        'ma_detail': {'alignment': '未知', 'sorted_mas': []},
        'consensus': {'direction': 'neutral', 'aggregate_score': 0.0, 'bullish_ratio': 0.0},
        'signals_active': 0, 'signals': [],
        'price': 0, 'change_5d': 0,
    }


def relative_strength(period_days: int = 60) -> dict:
    """
    计算11个板块ETF相对SPY的超额收益，按排名打分

    核心逻辑：
    - 计算每只ETF在 period_days 内的收益率
    - 减去 SPY 同期收益率 = 超额收益
    - 按超额收益排名：前3=强势(+1)、中5=中性(0)、后3=弱势(-1)

    Returns:
        {
            scores: {ticker: -1/0/1},
            rankings: [{ticker, sector, return_pct, excess, rank, label}],
        }
    """
    df = pd.read_csv(ETF_CSV, parse_dates=['Date'])

    # SPY 作为基准的同期收益
    spy = df[df['Ticker'] == 'SPY'].sort_values('Date')
    spy_latest = spy.iloc[-1]
    spy_start_idx = max(0, len(spy) - period_days - 1)
    spy_start = spy.iloc[spy_start_idx]
    spy_return = (spy_latest['Close'] - spy_start['Close']) / spy_start['Close'] * 100

    # 每只板块ETF的超额收益
    rankings = []
    for sector, ticker in SECTOR_ETF_MAP.items():
        etf_df = df[df['Ticker'] == ticker].sort_values('Date')
        if etf_df.empty or len(etf_df) < period_days:
            continue

        etf_latest = etf_df.iloc[-1]
        etf_start_idx = max(0, len(etf_df) - period_days - 1)
        etf_start = etf_df.iloc[etf_start_idx]
        etf_return = (etf_latest['Close'] - etf_start['Close']) / etf_start['Close'] * 100

        excess = round(etf_return - spy_return, 2)

        rankings.append({
            'ticker': ticker,
            'sector': sector,
            'return_pct': round(etf_return, 2),
            'excess': excess,
        })

    # 按超额收益排名
    rankings.sort(key=lambda x: x['excess'], reverse=True)

    # 排名赋值
    n = len(rankings)
    for i, r in enumerate(rankings):
        r['rank'] = i + 1
        if i < max(2, n // 4):        # 前25%
            r['label'] = '强势'
            r['rs_score'] = 1
        elif i >= n - max(2, n // 4):  # 后25%
            r['label'] = '弱势'
            r['rs_score'] = -1
        else:
            r['label'] = '中性'
            r['rs_score'] = 0

    scores = {r['sector']: r['rs_score'] for r in rankings}

    return {
        'scores': scores,
        'rankings': rankings,
        'spy_return': round(spy_return, 2),
        'period_days': period_days,
    }


def flow_judgment(trend_score: int, rs_score: int) -> dict:
    """
    资金流向判断矩阵：趋势 × 相对强度

               RS强势(+1)     RS中性(0)      RS弱势(-1)
    趋势多头(+1)   ✅流入+2     📈偏强+1       ⚠️ 高位派发0
    趋势中性(0)    📈偏强+1     ⏸️中性0        📉偏弱-1
    趋势空头(-1)   🔄空头反弹0   📉偏弱-1       ❌流出-2

    Returns:
        {score: -2~+2, label: str, flow_direction: 'inflow'/'outflow'/'neutral'/'divergence'}
    """
    matrix = {
        (1, 1):  (2, '资金确认流入', 'inflow'),
        (1, 0):  (1, '偏强', 'inflow'),
        (1, -1): (0, '高位分歧', 'divergence'),
        (0, 1):  (1, '偏强', 'inflow'),
        (0, 0):  (0, '方向不明', 'neutral'),
        (0, -1): (-1, '偏弱', 'outflow'),
        (-1, 1): (0, '空头反弹', 'divergence'),
        (-1, 0): (-1, '偏弱', 'outflow'),
        (-1, -1): (-2, '资金确认流出', 'outflow'),
    }

    score, label, flow = matrix.get((trend_score, rs_score), (0, '未知', 'neutral'))
    return {'score': score, 'label': label, 'flow_direction': flow}


def get_sector_snapshot() -> dict:
    """
    板块ETF完整快照：趋势 + 相对强度 + 资金流向 + 市场状态

    这是三维框架第二层的核心入口。

    Returns:
        {
            sectors: [{sector, etf, trend, rs, flow, score, price, change_5d}],
            top_sectors: [...],       # 强势板块（score >= 1）
            weak_sectors: [...],      # 弱势板块（score <= -1）
            rotation_signals: [...],  # 轮动信号
            market_state: str,        # 市场整体状态
            spy_trend: dict,          # SPY趋势参考
        }
    """
    # 1. 相对强度排名
    rs_result = relative_strength(period_days=60)
    rs_short = relative_strength(period_days=20)  # 短期也跑一次，用于检测轮动

    # 2. 每只ETF的趋势评分
    sectors = []
    spy_trend = None
    for sector, ticker in SECTOR_ETF_MAP.items():
        trend = etf_trend_score(ticker, period='6mo')
        rs = rs_result['scores'].get(sector, 0)
        rs_short_score = rs_short['scores'].get(sector, 0)

        flow = flow_judgment(trend['score'], rs)

        sectors.append({
            'sector': sector,
            'etf': ticker,
            'trend_score': trend['score'],
            'trend_label': trend['label'],
            'ma_alignment': trend['ma_detail']['alignment'],
            'consensus_direction': trend['consensus']['direction'],
            'rs_score': rs,
            'rs_short_score': rs_short_score,
            'flow_score': flow['score'],
            'flow_label': flow['label'],
            'flow_direction': flow['flow_direction'],
            'price': trend['price'],
            'change_5d': trend['change_5d'],
            'signals': trend['signals'],
        })

    # 3. SPY基准趋势
    try:
        spy_trend = etf_trend_score('SPY', period='6mo')
    except Exception:
        spy_trend = _empty_trend_result()

    # 4. 分类：强势/弱势/轮动信号
    top_sectors = [s for s in sectors if s['flow_score'] >= 1]
    weak_sectors = [s for s in sectors if s['flow_score'] <= -1]
    neutral_sectors = [s for s in sectors if -1 < s['flow_score'] < 1]

    # 排序：按 flow_score 降序
    sectors.sort(key=lambda x: x['flow_score'], reverse=True)

    # 5. 检测轮动信号
    rotation_signals = []
    for s in sectors:
        # 短期RS转强但长期RS中性 = 早期轮动信号
        if s['rs_short_score'] > s['rs_score']:
            rotation_signals.append({
                'sector': s['sector'],
                'etf': s['etf'],
                'type': 'early_rotation',
                'message': f"{s['sector']}短期RS转强，关注轮动早期信号",
            })
        # 趋势空头但RS强势 = 潜在反弹
        if s['trend_score'] == -1 and s['rs_score'] == 1:
            rotation_signals.append({
                'sector': s['sector'],
                'etf': s['etf'],
                'type': 'potential_reversal',
                'message': f"{s['sector']}空头趋势中RS强势，关注底部反弹",
            })
        # 趋势多头但RS弱势 = 高位派发风险
        if s['trend_score'] == 1 and s['rs_score'] == -1:
            rotation_signals.append({
                'sector': s['sector'],
                'etf': s['etf'],
                'type': 'distribution_risk',
                'message': f"{s['sector']}多头趋势但RS弱势，警惕高位派发",
            })

    # 6. 市场整体状态
    bullish_count = sum(1 for s in sectors if s['flow_score'] >= 1)
    bearish_count = sum(1 for s in sectors if s['flow_score'] <= -1)
    if bullish_count >= 6:
        market_state = 'risk_on'
    elif bearish_count >= 6:
        market_state = 'risk_off'
    elif bullish_count >= bearish_count + 2:
        market_state = 'cautious_bull'
    elif bearish_count >= bullish_count + 2:
        market_state = 'cautious_bear'
    else:
        market_state = 'neutral'

    return {
        'sectors': sectors,
        'top_sectors': top_sectors,
        'weak_sectors': weak_sectors,
        'neutral_sectors': neutral_sectors,
        'rotation_signals': rotation_signals,
        'market_state': market_state,
        'spy_trend': spy_trend,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
    }


# ============================================================
# CLI 测试
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("📊 板块ETF资金轮动分析")
    print("=" * 70)

    snapshot = get_sector_snapshot()

    print(f"\n市场状态: {snapshot['market_state']}")
    print(f"强势板块: {snapshot['bullish_count']}个 | 弱势板块: {snapshot['bearish_count']}个")

    print(f"\n{'板块':16s} {'ETF':5s} {'趋势':6s} {'RS':4s} {'资金流向':10s} {'评分':5s} {'5日涨跌':8s}")
    print("-" * 65)
    for s in snapshot['sectors']:
        print(f"{s['sector']:16s} {s['etf']:5s} {s['trend_label']:6s} "
              f"{'强势' if s['rs_score'] > 0 else '弱势' if s['rs_score'] < 0 else '中性':4s} "
              f"{s['flow_label']:10s} {s['flow_score']:3d}   {s['change_5d']:+6.2f}%")

    if snapshot['rotation_signals']:
        print(f"\n🔄 轮动信号 ({len(snapshot['rotation_signals'])}):")
        for sig in snapshot['rotation_signals']:
            print(f"  [{sig['type']}] {sig['message']}")

    print(f"\n📈 强势板块 (建议选股):")
    for s in snapshot['top_sectors']:
        print(f"  {s['sector']} ({s['etf']}): {s['flow_label']}, MA={s['ma_alignment']}")

    if snapshot['weak_sectors']:
        print(f"\n📉 弱势板块 (建议回避):")
        for s in snapshot['weak_sectors']:
            print(f"  {s['sector']} ({s['etf']}): {s['flow_label']}, MA={s['ma_alignment']}")

    # 相对强度排名详情
    print(f"\n🏆 相对强度排名 (60日超额收益 vs SPY):")
    rs = relative_strength(60)
    for r in rs['rankings']:
        print(f"  {r['rank']:2d}. {r['sector']:25s} ({r['ticker']}): {r['excess']:+6.2f}% ({r['label']})")
