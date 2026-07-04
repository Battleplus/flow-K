"""
全板块三维扫描 — 11个板块各选代表性股票
输出: outputs/stock_recommendations.json + 控制台表格
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.news_analysis import analyze_news
from src.sector_rotation import get_sector_snapshot
from src.screener import analyze_ticker
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
COMPANY_INFO = os.path.join(DATA_DIR, 'company_info.csv')

# 11个板块的代表性股票（手动挑选各板块龙头/高关注度个股）
SECTOR_REPS = {
    'Technology': [
        'NVDA','AMD','AVGO','AAPL','MSFT','GOOGL','META','AMAT','ADI','APH',
        'ACN','ADBE','CRWD','CDNS','ANET','CSCO','INTC','QCOM','TXN','LRCX',
        'ORCL','IBM','NOW','INTU','MU','PANW','SNPS','KLAC','MRVL','NXPI',
    ],
    'Industrials': [
        'GE','CAT','UNP','HON','RTX','UPS','LMT','DE','EMR','ETN',
        'BA','MMM','GEV','CP','NSC','WM','RSG','PWR','EME','CMI',
    ],
    'Healthcare': [
        'UNH','LLY','JNJ','ABBV','MRK','PFE','TMO','ABT','DHR','BMY',
        'AMGN','GILD','VRTX','REGN','ISRG','CI','HCA','MDT','ZTS','SYK',
    ],
    'Financials': [
        'JPM','BAC','WFC','GS','MS','BLK','C','MET','PRU','AXP',
        'SCHW','SPGI','MMC','PNC','USB','TFC','COF','AIG','ALL','TRV',
    ],
    'Consumer Discretionary': [
        'AMZN','TSLA','HD','MCD','NKE','SBUX','LOW','F','GM','BKNG',
        'TJX','ABNB','TGT','CMG','ROST','EBAY','DPZ','YUM','LEN','DHI',
    ],
    'Consumer Staples': [
        'PG','KO','PEP','COST','WMT','PM','MO','MDLZ','CL','TGT',
        'KMB','GIS','K','STZ','HSY','CAG','CHD','SJM','MKC','TSN',
    ],
    'Utilities': [
        'NEE','DUK','SO','D','AEP','EXC','SRE','XEL','WEC','ED',
        'AWK','AMT','PLD','CCI','EQIX','PSA','DLR','SPG','O','VTR',
    ],
    'Materials': [
        'LIN','APD','SHW','FCX','NEM','DOW','DD','NUE','PPG','ECL',
        'VMC','MLM','IFF','FNV','AA','CF','MOS','CE','EMN','ALB',
    ],
    'Real Estate': [
        'PLD','AMT','CCI','EQIX','PSA','DLR','SPG','O','WELL','AVB',
        'EQR','VTR','INVH','EXR','CSGP','ARE','MAA','UDR','BXP','KRC',
    ],
    'Energy': [
        'XOM','CVX','COP','SLB','EOG','PSX','MPC','VLO','HES','OXY',
        'PXD','DVN','FANG','CTRA','HFC','WMB','KMI','OKE','TRGP','WES',
    ],
    'Communication Services': [
        'GOOGL','META','NFLX','DIS','CMCSA','T','VZ','TMUS','CHTR','TME',
        'TTWO','EA','LYV','WBD','FOX','FOXA','NWSA','NWS','DISH','SPGI',
    ],
}

# 标准化板块名映射
SECTOR_NAME_MAP = {
    'Information Technology': 'Technology',
    'Health Care': 'Healthcare',
}

def normalize_sector(s):
    return SECTOR_NAME_MAP.get(s, s)

# 多条新闻场景
NEWS_SCENARIOS = [
    'AI semiconductor technology artificial intelligence chips data center GPU',
    'Federal Reserve interest rate decision inflation CPI consumer prices',
    'oil price surge energy crisis OPEC production cut',
    'healthcare drug FDA approval pharmaceutical biotech breakthrough',
    'consumer spending retail sales strong economy growth',
]

def stock_score_to_3d(direction, aggregate_score):
    direction_map = {
        'strong_bullish': 2.0, 'bullish': 1.0, 'neutral': 0.0,
        'bearish': -1.0, 'strong_bearish': -2.0,
    }
    base = direction_map.get(direction, 0.0)
    adj = max(-0.5, min(0.5, aggregate_score * 0.2))
    return round(max(-2.0, min(2.0, base + adj)), 1)


def main():
    print('=' * 80)
    print('  FLOW AI — 全板块三维扫描')
    print('=' * 80)

    # ======== 消息面（多场景） ========
    print('\n[1/3] 消息面分析（多场景）...')
    all_news_scores = {}  # sector -> max score across scenarios
    news_details = []
    for scenario in NEWS_SCENARIOS:
        news = analyze_news(scenario)
        for sector, score in news['sector_scores'].items():
            if sector not in all_news_scores or score > all_news_scores[sector]:
                all_news_scores[sector] = score
        news_details.append({
            'text': scenario[:60] + '...' if len(scenario) > 60 else scenario,
            'scores': news['sector_scores'],
        })
        print(f'  "{scenario[:50]}..."')
        for s, v in sorted(news['sector_scores'].items(), key=lambda x: -x[1])[:3]:
            if v >= 0.5:
                print(f'    {s}: {v:+.1f}')

    # ======== 板块ETF轮动 ========
    print('\n[2/3] 板块ETF轮动快照...')
    snap = get_sector_snapshot()
    sector_flow = {}  # sector -> flow_score
    sector_info = {}
    for s in snap['sectors']:
        sector_flow[s['sector']] = s['flow_score']
        sector_info[s['sector']] = s
        label = s['flow_label']
        print(f"  {s['sector']:25s} ETF={s['etf']} flow={s['flow_score']:+d}({label}) rs60={s['rs_score']:+d} rs20={s['rs_short_score']:+d}")

    # ======== 个股扫描 ========
    print('\n[3/3] 个股K线扫描（11个板块，代表股）...')
    all_results = []
    total = sum(len(v) for v in SECTOR_REPS.values())
    idx = 0

    for sector, tickers in SECTOR_REPS.items():
        news_score = all_news_scores.get(sector, 0.0)
        flow_score = sector_flow.get(sector, 0)
        sector_score = news_score * 0.4 + flow_score * 0.6
        print(f'\n  --- {sector} ({len(tickers)}只) news={news_score:+.1f} flow={flow_score:+d} sector_score={sector_score:+.1f} ---')

        for ticker in tickers:
            idx += 1
            t0 = time.time()
            try:
                info = analyze_ticker(ticker, period='6mo')
                if info is None:
                    continue
            except Exception:
                continue

            stock_3d = stock_score_to_3d(info['consensus_direction'], info['aggregate_score'])
            final_score = news_score * 0.25 + sector_score * 0.35 + stock_3d * 0.4
            final_score = round(max(-3.0, min(3.0, final_score)), 1)

            # 获取公司名
            company_name = ticker
            try:
                df_ci = pd.read_csv(COMPANY_INFO)
                row = df_ci[df_ci['Symbol'] == ticker]
                if not row.empty:
                    company_name = row.iloc[0]['Company']
            except Exception:
                pass

            all_results.append({
                'ticker': ticker,
                'name': company_name,
                'sector': sector,
                'news_score': round(news_score, 1),
                'flow_score': flow_score,
                'sector_score': round(sector_score, 1),
                'stock_score': stock_3d,
                'final_score': final_score,
                'direction': info['consensus_direction'],
                'consensus': info['consensus_text'],
                'sharpe': round(info['avg_sharpe_top5'], 2) if info['avg_sharpe_top5'] else 0,
                'pf_return': round(info['pf_return'], 1) if info['pf_return'] else 0,
                'close': round(info['close'], 2) if info['close'] else 0,
                'signal_net': info['signal_net'],
                'market_regime': info.get('market_regime', ''),
            })
            dt = time.time() - t0
            mark = '★' if final_score >= 1.0 else ('●' if final_score >= 0.5 else '')
            print(f'    {idx:3d}/{total} {ticker:5s} ({company_name[:20]:20s}) stock={stock_3d:+.1f} final={final_score:+.1f} {info["consensus_direction"]:15s} {mark}')

    # ======== 排序 ========
    all_results.sort(key=lambda x: x['final_score'], reverse=True)

    # 每板块TOP3
    sector_tops = {}
    for r in all_results:
        s = r['sector']
        if s not in sector_tops:
            sector_tops[s] = []
        if len(sector_tops[s]) < 3:
            sector_tops[s].append(r)

    # 综合TOP20
    top20 = all_results[:20]

    # ======== 输出JSON ========
    output = {
        'scan_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'market_state': snap.get('market_state', 'neutral'),
        'spy_price': snap.get('spy_trend', {}).get('price', 0) if isinstance(snap.get('spy_trend'), dict) else 0,
        'news_scenarios': news_details,
        'sector_snapshot': [{
            'sector': s['sector'],
            'etf': s['etf'],
            'flow_score': s['flow_score'],
            'flow_label': s['flow_label'],
            'rs_score': s['rs_score'],
            'rs_short_score': s['rs_short_score'],
            'price': round(s['price'], 2),
            'change_5d': round(float(s['change_5d']), 2),
            'signals': s.get('signals', []),
        } for s in snap['sectors']],
        'sector_tops': {k: v for k, v in sector_tops.items()},
        'top20': top20,
        'all_results': all_results,
        'total_scanned': len(all_results),
    }

    out_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'stock_recommendations.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n✅ JSON已保存: {out_path}')

    # ======== 控制台汇总 ========
    print('\n' + '=' * 110)
    print('  综合推荐 TOP 20')
    print('=' * 110)
    print(f'{"排名":4s} {"代码":6s} {"公司":22s} {"板块":22s} {"消息面":6s} {"板块分":6s} {"个股分":6s} {"综合分":6s} {"共识方向":16s} {"夏普":6s} {"回测收益":8s}')
    print('-' * 110)
    for i, r in enumerate(top20):
        print(f'{i+1:3d}.  {r["ticker"]:5s} {r["name"][:20]:20s} {r["sector"][:20]:20s} {r["news_score"]:+5.1f}  {r["sector_score"]:+5.1f}  {r["stock_score"]:+5.1f}  {r["final_score"]:+5.1f}  {r["direction"]:16s} {r["sharpe"]:5.2f}  {r["pf_return"]:7.1f}%')

    print('\n' + '=' * 110)
    print('  各板块代表股 TOP 3')
    print('=' * 110)
    for sector in SECTOR_REPS.keys():
        tops = sector_tops.get(sector, [])
        print(f'\n  【{sector}】')
        for i, r in enumerate(tops):
            print(f'    {i+1}. {r["ticker"]:5s} ({r["name"][:20]:20s}) 综合={r["final_score"]:+.1f} 个股={r["stock_score"]:+.1f} {r["direction"]:16s} 夏普={r["sharpe"]:.2f} 回测={r["pf_return"]:.1f}%')

    avg = sum(r['final_score'] for r in top20[:10]) / min(len(top20), 10)
    if avg >= 1.5: action = '强烈看多'
    elif avg >= 0.5: action = '偏多'
    elif avg >= -0.5: action = '观望'
    elif avg >= -1.5: action = '偏空'
    else: action = '强烈看空'
    print(f'\n📊 综合判定: {action} (TOP10均值 {avg:+.1f}) | 共扫描 {len(all_results)} 只股票')


if __name__ == '__main__':
    main()
