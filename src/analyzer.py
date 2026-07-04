"""K 线趋势斜率信号分析工具 + 策略引擎 - CLI 入口

用法:
    python src/analyzer.py NVDA                    # 默认 1 年日线
    python src/analyzer.py NVDA --period 6mo       # 半年
    python src/analyzer.py NVDA --strategy full_monty # 回测指定策略
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_data
from src.indicators import trend_analysis
from src.patterns import add_indicators, detect_all, signal_summary
from src.chart import plot_kline_trend


def analyze(ticker: str, period: str = "1y", strategy: str | None = None, mode: str = "multi"):
    """主分析函数：拉取数据 → 计算指标 → 信号检测 → 策略回测 → 绘图"""
    print(f"\n{'='*60}")
    print(f"  K 线趋势斜率信号分析: {ticker}")
    print(f"{'='*60}\n")

    print("[1/5] 拉取数据...")
    df = fetch_data(ticker, period=period)
    print(f"  数据范围: {df['Date'].min()} ~ {df['Date'].max()}, {len(df)} 条")

    print("[2/5] 计算趋势指标...")
    analysis = trend_analysis(df)

    print(f"\n  📊 整体趋势: {analysis['overall_trend']}")
    print(f"  📐 均线排列: {analysis['ma_alignment']['alignment']}")
    print(f"     {analysis['ma_alignment']['detail']}")
    print(f"\n  均线斜率:")
    for name, info in analysis["slopes"].items():
        symbol = "↑" if info["direction"] == "上升" else "↓" if info["direction"] == "下降" else "→"
        print(f"    {name}: {info['slope']}% {symbol} {info['direction']} (值: {info['value']})")

    sr = analysis["support_resistance"]
    print(f"\n  📍 支撑位: {sr['support']} (距当前 {sr['pct_to_support']}%)")
    print(f"  📍 压力位: {sr['resistance']} (距当前 +{sr['pct_to_resistance']}%)")

    crosses = analysis["crosses"]
    if crosses:
        print(f"\n  近期均线交叉:")
        for c in crosses[-5:]:
            print(f"    {c['date']}: {c['type']} ({c['ma1']} x {c['ma2']})")

    print("\n[3/5] 检测趋势线/斜率/曲线/形态信号...")
    df = add_indicators(df)
    df = detect_all(df)
    sig = signal_summary(df)

    print(f"\n  ⚡ 信号评分: {sig['score']}  →  {sig['verdict']}")
    print(f"  偏多: {sig['bullish_count']}  | 偏空: {sig['bearish_count']}  | 中性: {sig['neutral_count']}")
    if sig["active_signals"]:
        print(f"\n  近期活跃信号:")
        for s in sig["active_signals"]:
            dir_emoji = "🔴" if s['direction'] == 'bullish' else "🟢" if s['direction'] == 'bearish' else "⚪"
            print(f"    {dir_emoji} {s['name']} (强度 {s['strength']}) - {s['last_date']}")

    print("\n[4/5] 策略引擎回测...")
    from src.factors import add_all_factors
    from src.strategies import (
        STRATEGIES, backtest_all, backtest_strategy,
        rank_strategies, get_latest_consensus, get_strategy_summary,
    )
    df = add_all_factors(df)
    if strategy:
        if strategy not in STRATEGIES:
            print(f"  错误: 未知策略 {strategy}")
            print(f"  可用策略: {', '.join(STRATEGIES.keys())}")
            sys.exit(1)
        r = backtest_strategy(df, strategy)
        print(f"\n  📋 策略: {r.strategy_name} ({STRATEGIES[strategy].category})")
        print(f"     总收益: {r.total_return*100:.2f}%  | 年化: {r.annual_return*100:.2f}%")
        print(f"     夏普: {r.sharpe_ratio:.2f}  | 最大回撤: {r.max_drawdown*100:.1f}%")
        print(f"     胜率: {r.win_rate*100:.0f}%  | 交易次数: {r.total_trades}")
    else:
        results = backtest_all(df, min_trades=1)
        ranked = rank_strategies(results, "sharpe_ratio")
        print(f"\n  回测完成, {len(results)} 个有效策略")
        print(f"\n  🏆 按夏普比率 TOP 5:")
        for i, (sid, r, _) in enumerate(ranked[:5]):
            print(f"    {i+1}. {r.strategy_name:20s} 收益{r.total_return*100:6.1f}%  夏普{r.sharpe_ratio:5.2f}  胜率{r.win_rate*100:3.0f}%  交易{r.total_trades}次")

    cons = get_latest_consensus(df)
    print(f"\n  🗳 多策略共识: {cons['consensus']} (评分: {cons['aggregate_score']:.0f})")
    print(f"     偏多策略比例: {cons['bullish_ratio']*100:.0f}%")
    if cons['active_strategies']:
        print(f"     当前活跃信号:")
        for sid, info in cons['active_strategies'].items():
            sig_text = "买入" if info['signal'] > 0 else "卖出"
            print(f"       [{info['category']}] {info['name']}: {sig_text}")

    print("\n[5/5] 生成图表...")
    chart_path = plot_kline_trend(df, ticker, analysis, sig, mode=mode)
    print(f"  图表: {chart_path}")
    print(f"\n  ✅ 分析完成！\n")
    return chart_path


def main():
    parser = argparse.ArgumentParser(
        description="K 线趋势斜率信号 + 多因子策略回测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/analyzer.py NVDA
  python src/analyzer.py NVDA --period 6mo
  python src/analyzer.py NVDA --strategy ma_ribbon
  python src/analyzer.py TSLA --mode single
        """,
    )
    parser.add_argument("ticker", help="股票代码 (美股: AAPL, 港股: 0700.HK)")
    parser.add_argument("--period", "-p", default="1y", help="数据周期 (1y/6mo/3mo/1mo)")
    parser.add_argument("--strategy", "-s", default=None, help="指定策略ID (默认回测全部)")
    parser.add_argument("--mode", default="multi", help="图表模式 (multi 或 single)")
    parser.add_argument("--mas", "-m", default=None, help="均线周期，逗号分隔 (默认: 5,10,20,60)")

    args = parser.parse_args()

    try:
        analyze(args.ticker, period=args.period, strategy=args.strategy, mode=args.mode)
    except ValueError as e:
        print(f"  错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
