"""主分析入口 - K 线策略研究工具

用法:
    python src/analyzer.py NVDA                        # 默认分析单只股票
    python src/analyzer.py --ticker AAPL                # 命名参数方式
    python src/analyzer.py --list                        # 列出所有可用股票
    python src/analyzer.py --batch AAPL,MSFT,GOOGL      # 批量分析多只股票
    python src/analyzer.py NVDA --signals big_bull_breakout,long_lower_shadow
    python src/analyzer.py NVDA --lookback 1             # 只看最近 1 年数据
    python src/analyzer.py NVDA --no-chart               # 不生成图表
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw, load_ticker, list_tickers
from src.patterns import add_indicators, detect_all, SIGNALS
from src.stats import compute_forward_returns, all_signal_stats
from src.chart import plot_kline_with_signals, plot_forward_distribution
from src.report import generate_report


def analyze(
    ticker: str,
    lookback_years: float = 3.0,
    selected_signals: list[str] | None = None,
    generate_charts: bool = True,
):
    """主分析函数：加载数据 -> 计算指标 -> 检测信号 -> 统计 -> 生成报告"""
    print(f"\n{'='*60}")
    print(f"  K 线策略分析: {ticker}")
    print(f"{'='*60}\n")

    # 确定使用哪些信号
    if selected_signals:
        active_signals = {k: v for k, v in SIGNALS.items() if k in selected_signals}
        if not active_signals:
            print(f"  警告: 指定的信号名无效，将使用全部信号。可用: {', '.join(SIGNALS.keys())}")
            active_signals = SIGNALS
    else:
        active_signals = SIGNALS

    # 1. 加载数据
    print("[1/5] 加载数据...")
    raw = load_raw()
    df = load_ticker(raw, ticker)
    df = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=int(lookback_years * 365))]
    print(f"  数据范围: {df['Date'].min().date()} ~ {df['Date'].max().date()}, {len(df)} 条")

    # 2. 计算指标
    print("[2/5] 计算技术指标...")
    df = add_indicators(df)
    df = detect_all(df)
    df = compute_forward_returns(df)

    # 3. 统计
    print("[3/5] 统计历史信号表现...")
    all_stats = all_signal_stats(df)
    stats = {k: v for k, v in all_stats.items() if k in active_signals}

    # 4. 检查当前信号
    print("[4/5] 检查当前信号...")
    latest = df.iloc[-1]
    current_signals = []
    for key in active_signals:
        col = f"signal_{key}"
        if col in df.columns and latest[col] == 1:
            current_signals.append(key)

    if current_signals:
        print(f"  当前触发信号: {', '.join(current_signals)}")
    else:
        print("  当前未触发任何信号")

    # 5. 绘图
    chart_path = ""
    dist_charts = []
    if generate_charts:
        print("[5/5] 生成图表...")
        signal_cols = [f"signal_{k}" for k in active_signals]
        chart_path = plot_kline_with_signals(df, ticker, signal_cols=signal_cols)

        for key in (current_signals if current_signals else list(active_signals.keys())):
            p = plot_forward_distribution(df, f"signal_{key}", ticker, forward_n=5)
            if p:
                dist_charts.append(p)
    else:
        print("[5/5] 跳过图表生成")

    # 6. 报告
    report_path = generate_report(ticker, df, stats, current_signals, chart_path, dist_charts)

    print(f"\n  分析完成！")
    if chart_path:
        print(f"  图表: {chart_path}")
    print(f"  报告: {report_path}")
    print()

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="K 线策略研究工具 - 统计型 K 线信号回测分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/analyzer.py NVDA
  python src/analyzer.py --ticker AAPL --signals big_bull_breakout,support_bounce
  python src/analyzer.py --batch AAPL,MSFT,GOOGL
  python src/analyzer.py --list
        """,
    )

    parser.add_argument(
        "ticker_positional",
        nargs="?",
        default=None,
        help="股票代码位置参数（如 NVDA, AAPL）",
    )
    parser.add_argument(
        "--ticker", "-t",
        default=None,
        help="股票代码命名参数",
    )
    parser.add_argument(
        "--batch", "-b",
        default=None,
        help="批量分析多只股票，逗号分隔（如 AAPL,MSFT,GOOGL）",
    )
    parser.add_argument(
        "--signals", "-s",
        default=None,
        help="指定使用的信号，逗号分隔。可用: big_bull_breakout, support_bounce, long_lower_shadow, reversal_after_decline",
    )
    parser.add_argument(
        "--lookback", "-l",
        type=float,
        default=3.0,
        help="回看年数（默认 3.0）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的股票代码",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="跳过图表生成，仅输出报告",
    )

    args = parser.parse_args()

    # --list
    if args.list:
        raw = load_raw()
        tickers = list_tickers(raw)
        print(f"\n可用股票代码（共 {len(tickers)} 只）:\n")
        for i, t in enumerate(tickers):
            print(f"  {t:<8}", end="")
            if (i + 1) % 8 == 0:
                print()
        print("\n")
        return

    # 确定 ticker 列表
    tickers = []
    if args.batch:
        tickers = [t.strip().upper() for t in args.batch.split(",") if t.strip()]
    else:
        single = args.ticker or args.ticker_positional
        if not single:
            parser.print_help()
            print("\n错误: 请指定股票代码（位置参数 / --ticker / --batch / --list）")
            sys.exit(1)
        tickers = [single.strip().upper()]

    # 解析信号
    selected_signals = None
    if args.signals:
        selected_signals = [s.strip() for s in args.signals.split(",") if s.strip()]

    # 逐个分析
    raw = load_raw()
    for t in tickers:
        try:
            analyze(
                ticker=t,
                lookback_years=args.lookback,
                selected_signals=selected_signals,
                generate_charts=not args.no_chart,
            )
        except ValueError as e:
            print(f"  跳过 {t}: {e}")
        except Exception as e:
            print(f"  分析 {t} 时出错: {e}")


if __name__ == "__main__":
    main()
