"""K 线趋势斜率分析工具 - CLI 入口

用法:
    python src/analyzer.py NVDA                    # 默认 1 年日线
    python src/analyzer.py NVDA --period 6mo       # 半年
    python src/analyzer.py 0700.HK                 # 港股
    python src/analyzer.py AAPL --mas 5,10,20,60   # 自定义均线周期
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_data
from src.indicators import trend_analysis
from src.chart import plot_kline_trend


def analyze(ticker: str, period: str = "1y", mas: list[int] | None = None):
    """主分析函数：拉取数据 → 计算指标 → 绘图"""
    print(f"\n{'='*60}")
    print(f"  K 线趋势斜率分析: {ticker}")
    print(f"{'='*60}\n")

    print("[1/3] 拉取数据...")
    df = fetch_data(ticker, period=period)
    print(f"  数据范围: {df['Date'].min()} ~ {df['Date'].max()}, {len(df)} 条")

    print("[2/3] 计算趋势指标...")
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

    print("\n[3/3] 生成图表...")
    chart_path = plot_kline_trend(df, ticker, analysis)
    print(f"  图表: {chart_path}")
    print(f"\n  ✅ 分析完成！\n")
    return chart_path


def main():
    parser = argparse.ArgumentParser(
        description="K 线趋势斜率分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/analyzer.py NVDA
  python src/analyzer.py NVDA --period 6mo
  python src/analyzer.py 0700.HK
  python src/analyzer.py AAPL --mas 5,10,20,60
        """,
    )
    parser.add_argument("ticker", help="股票代码 (美股: AAPL, 港股: 0700.HK)")
    parser.add_argument("--period", "-p", default="1y", help="数据周期 (1y/6mo/3mo/1mo)")
    parser.add_argument("--mas", "-m", default=None, help="均线周期，逗号分隔 (默认: 5,10,20,60)")

    args = parser.parse_args()

    mas = None
    if args.mas:
        mas = [int(m.strip()) for m in args.mas.split(",") if m.strip()]

    try:
        analyze(args.ticker, period=args.period, mas=mas)
    except ValueError as e:
        print(f"  错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
