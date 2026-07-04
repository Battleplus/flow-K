"""报告生成模块 - 输出 Markdown 分析报告"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from src.patterns import SIGNALS

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _judge_signal(stats: dict, min_samples: int = 10) -> str:
    """根据统计结果给出偏多/中性/偏空判断"""
    fwd = stats.get("forward", {})
    d5 = fwd.get("5d", {})
    total = stats.get("total_occurrences", 0)

    if total < min_samples:
        return "样本不足"

    win = d5.get("win_rate", 50)
    avg = d5.get("avg_ret", 0)

    if win >= 60 and avg > 0:
        return "偏多"
    elif win <= 45 and avg < 0:
        return "偏空"
    else:
        return "中性"


def _overall_judgment(stats: dict) -> dict:
    """综合所有信号给出整体判断"""
    bullish = 0
    bearish = 0
    total_samples = 0

    for key, s in stats.items():
        fwd = s.get("forward", {}).get("5d", {})
        total = s.get("total_occurrences", 0)
        total_samples += total

        if total >= 10:
            win = fwd.get("win_rate", 50)
            avg = fwd.get("avg_ret", 0)
            if win >= 60 and avg > 0:
                bullish += 1
            elif win <= 45 and avg < 0:
                bearish += 1

    if bullish > bearish:
        verdict = "偏多 —— 多个信号历史上偏多的统计显著性较高"
    elif bearish > bullish:
        verdict = "偏空 —— 多个信号历史上偏空的统计显著性较高"
    else:
        verdict = "中性 —— 各信号统计结果混合，无明显方向性"

    return {
        "bullish_count": bullish,
        "bearish_count": bearish,
        "total_samples": total_samples,
        "verdict": verdict,
    }


def generate_report(
    ticker: str,
    df,
    stats: dict,
    current_signals: list[str],
    chart_path: str,
    dist_charts: list[str],
) -> str:
    """生成完整的 Markdown 报告"""
    latest = df.iloc[-1]
    lines = []

    lines.append(f"# K 线策略分析报告: {ticker}")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"**数据范围**: {df['Date'].min().date()} ~ {df['Date'].max().date()}，共 {len(df)} 条")
    lines.append("")
    lines.append(f"**最近收盘价**: {latest['Adj Close']:.2f}")
    lines.append("")

    # 当前信号
    lines.append("## 当前信号")
    lines.append("")
    if current_signals:
        for key in current_signals:
            name = SIGNALS[key][0]
            lines.append(f"- **{name}** ({key})")
        lines.append("")
    else:
        lines.append("当前未触发任何 K 线信号。")
        lines.append("")

    # 综合判断
    overall = _overall_judgment(stats)
    lines.append("## 综合判断")
    lines.append("")
    lines.append(f"> **{overall['verdict']}**")
    lines.append("")
    lines.append(f"- 偏多信号数: {overall['bullish_count']}")
    lines.append(f"- 偏空信号数: {overall['bearish_count']}")
    lines.append(f"- 历史信号总样本数: {overall['total_samples']}")
    lines.append("")
    lines.append("> *综合判断基于各信号的历史 5 日胜率与平均收益统计，样本充足（≥10次）的信号纳入计算。*")
    lines.append("")

    # 最近行情快照
    lines.append("## 最近行情快照")
    lines.append("")
    lines.append("| 日期 | 收盘 | MA5 | MA20 | MA60 | 成交量 |")
    lines.append("|------|------|-----|------|------|--------|")
    for _, row in df.tail(5).iterrows():
        d = row["Date"].strftime("%Y-%m-%d")
        c = f"{row['Close']:.2f}"
        m5 = f"{row['ma5']:.2f}" if not pd.isna(row.get("ma5")) else "-"
        m20 = f"{row['ma20']:.2f}" if not pd.isna(row.get("ma20")) else "-"
        m60 = f"{row['ma60']:.2f}" if not pd.isna(row.get("ma60")) else "-"
        v = f"{row['Volume']:,.0f}"
        lines.append(f"| {d} | {c} | {m5} | {m20} | {m60} | {v} |")
    lines.append("")

    # 各信号统计
    lines.append("## 信号历史统计")
    lines.append("")
    for key, (name, _) in SIGNALS.items():
        s = stats.get(key, {})
        total = s.get("total_occurrences", 0)
        judge = _judge_signal(s)
        lines.append(f"### {name} ({key})")
        lines.append("")
        lines.append(f"- **历史出现次数**: {total}")
        lines.append(f"- **当前判断**: {judge}")
        lines.append("")
        fwd = s.get("forward", {})
        if fwd:
            lines.append("| 未来 | 平均收益 | 胜率 | 最大回撤 | 最佳 | 最差 | 中位数 |")
            lines.append("|------|----------|------|----------|------|------|--------|")
            for period_key in ["1d", "3d", "5d", "10d", "20d"]:
                if period_key in fwd:
                    fd = fwd[period_key]
                    lines.append(
                        f"| {period_key} | {fd['avg_ret']}% | {fd['win_rate']}% | "
                        f"{fd['max_dd']}% | {fd['best']}% | {fd['worst']}% | {fd['median']}% |"
                    )
            lines.append("")
        else:
            lines.append("样本不足，无法统计。")
            lines.append("")

    # 图表
    lines.append("## 图表")
    lines.append("")
    if chart_path:
        lines.append(f"![K线图]({chart_path})")
        lines.append("")
    for cp in dist_charts:
        if cp:
            lines.append(f"![收益分布]({cp})")
            lines.append("")

    report_text = "\n".join(lines)
    fpath = OUTPUT_DIR / f"report_{ticker}.md"
    fpath.write_text(report_text, encoding="utf-8")
    return str(fpath)
