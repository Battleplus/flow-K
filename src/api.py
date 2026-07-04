"""Flask API + 仪表盘服务端 — 集成因子/多面板/形态搜索/策略引擎"""

from flask import Flask, jsonify, request, send_from_directory, send_file
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_data
from src.indicators import trend_analysis
from src.patterns import add_indicators, detect_all, signal_summary, get_active_signals
from src.chart import plot_kline_trend
from src.factors import add_all_factors

app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="")


@app.route("/")
def index():
    return send_from_directory(str(ROOT / "web"), "index.html")


@app.route("/api/chart")
def api_chart():
    path = request.args.get("path", "")
    p = Path(path)
    if not p.exists():
        return ("Not found", 404)
    return send_file(str(p), mimetype="image/png")


@app.route("/api/analyze/<ticker>")
def api_analyze(ticker):
    period = request.args.get("period", "1y")
    mode = request.args.get("mode", "multi")
    try:
        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)
        analysis = trend_analysis(df)
        df_sig = detect_all(df)
        sig = signal_summary(df_sig)
        active = get_active_signals(df_sig, recent_days=5)
        chart_path = plot_kline_trend(df, ticker, analysis, sig, mode=mode)

        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "mode": mode,
            "data_start": str(df["Date"].min()),
            "data_end": str(df["Date"].max()),
            "records": len(df),
            "factor_count": len(df.columns),
            "analysis": analysis,
            "signals": {
                "score": sig["score"],
                "verdict": sig["verdict"],
                "bullish_count": sig["bullish_count"],
                "bearish_count": sig["bearish_count"],
                "neutral_count": sig["neutral_count"],
                "active": active,
            },
            "chart_path": chart_path,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/signals/<ticker>")
def api_signals(ticker):
    period = request.args.get("period", "1y")
    try:
        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)
        df = detect_all(df)
        sig = signal_summary(df)
        return jsonify({"ticker": ticker.upper(), "period": period, "signals": sig})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 策略引擎 API ──

@app.route("/api/strategies")
def api_strategies_list():
    """列出所有可用策略"""
    from src.strategies import STRATEGIES
    result = {}
    for sid, s in STRATEGIES.items():
        result[sid] = {
            "name": s.name,
            "category": s.category,
            "description": s.description,
            "factors_used": s.factors_used,
            "stop_loss_pct": s.stop_loss_pct,
            "take_profit_pct": s.take_profit_pct,
        }
    return jsonify({"count": len(result), "strategies": result})


@app.route("/api/backtest/<ticker>")
def api_backtest(ticker):
    """
    对指定股票回测所有策略
    参数:
      strategy: 指定策略ID (可选, 不传则回测全部)
      period:   数据周期 (默认1y)
      capital:  初始资金 (默认100000)
    """
    period = request.args.get("period", "1y")
    strategy_id = request.args.get("strategy", None)
    capital = float(request.args.get("capital", 100000))

    try:
        from src.strategies import STRATEGIES, backtest_strategy, backtest_all, get_strategy_summary

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)

        if strategy_id:
            if strategy_id not in STRATEGIES:
                return jsonify({"error": f"未知策略: {strategy_id}"}), 400
            r = backtest_strategy(df, strategy_id, initial_capital=capital)
            result = {
                "ticker": ticker.upper(),
                "period": period,
                "strategy_id": strategy_id,
                "strategy_name": r.strategy_name,
                "total_return": round(r.total_return * 100, 2),
                "annual_return": round(r.annual_return * 100, 2),
                "sharpe_ratio": round(r.sharpe_ratio, 2),
                "max_drawdown": round(r.max_drawdown * 100, 2),
                "win_rate": round(r.win_rate * 100, 1),
                "total_trades": r.total_trades,
                "avg_return_per_trade": round(r.avg_return_per_trade * 100, 2),
                "avg_hold_days": round(r.avg_hold_days, 0),
                "profit_factor": round(r.profit_factor, 2),
                # 最近20天的信号
                "recent_signals": r.signal_series.iloc[-20:].tolist(),
            }
        else:
            results = backtest_all(df, initial_capital=capital, min_trades=1)
            summary_df = get_strategy_summary(results)
            result = {
                "ticker": ticker.upper(),
                "period": period,
                "strategies": {},
                "ranked_by_sharpe": [],
            }
            for _, row in summary_df.iterrows():
                sid = row["策略ID"]
                result["strategies"][sid] = {
                    "name": row["策略名称"],
                    "category": row["分类"],
                    "total_return": row["总收益率"],
                    "annual_return": row["年化收益"],
                    "sharpe_ratio": row["夏普比率"],
                    "max_drawdown": row["最大回撤"],
                    "win_rate": row["胜率"],
                    "total_trades": row["交易次数"],
                    "avg_return_per_trade": row["平均收益"],
                    "avg_hold_days": row["平均持仓"],
                    "profit_factor": row["盈亏比"],
                }
            from src.strategies import rank_strategies
            ranked = rank_strategies(results, "sharpe_ratio")
            result["ranked_by_sharpe"] = [sid for sid, _, _ in ranked]

        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/consensus/<ticker>")
def api_consensus(ticker):
    """
    获取多策略共识信号
    返回所有策略的最新信号 + 综合评分
    """
    period = request.args.get("period", "1y")
    try:
        from src.strategies import get_latest_consensus, aggregate_signals

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)

        consensus = get_latest_consensus(df)

        # 额外：获取最近20天的共识分数趋势
        agg = aggregate_signals(df)
        recent = agg.iloc[-60:][["aggregate_score", "bullish_ratio"]].copy()
        recent["date"] = df["Date"].iloc[-60:].values

        return jsonify({
            "ticker": ticker.upper(),
            "consensus": consensus,
            "trend": [
                {"date": str(row["date"])[:10], "score": float(row["aggregate_score"]), "ratio": float(row["bullish_ratio"])}
                for _, row in recent.iterrows()
            ],
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/strategy_chart/<ticker>")
def api_strategy_chart(ticker):
    """
    生成策略组合分析图
    包含: K线 + 多策略信号标注 + 权益曲线
    """
    period = request.args.get("period", "1y")
    strategy_id = request.args.get("strategy", None)

    try:
        from src.strategies import STRATEGIES, backtest_strategy, aggregate_signals
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as _fm
        import mplfinance as mpf
        import numpy as np
        import pandas as pd

        # 中文字体设置
        _FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
        _fm.fontManager.addfont(_FONT_PATH)
        _cn_font = _fm.FontProperties(fname=_FONT_PATH)
        _cn_font_name = _cn_font.get_name()
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_cn_font_name, "Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)

        # 生成K线底图
        df_plot = df.copy()
        df_plot["Date_dt"] = pd.to_datetime(df_plot["Date"])
        df_plot = df_plot.set_index("Date_dt")
        n = len(df_plot)

        # 选择标注的策略: 默认选夏普最高的 5 个
        if strategy_id and strategy_id in STRATEGIES:
            selected = [strategy_id]
        else:
            from src.strategies import backtest_all
            results = backtest_all(df, min_trades=1)
            ranked = sorted(results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
            selected = [sid for sid, _ in ranked[:5]]

        # 生成信号
        signals = {}
        for sid in selected:
            try:
                signals[sid] = STRATEGIES[sid].generate(df)
            except Exception:
                signals[sid] = pd.Series(0, index=df.index)

        # 构建 addplot
        apds = []
        # 均线
        for col, color in [("ma5", "#ec2c2c"), ("ma10", "#2196f3"), ("ma20", "#9c27b0"), ("ma60", "#4caf50")]:
            if col in df_plot.columns:
                apds.append(mpf.make_addplot(df_plot[col], color=color, width=0.8))
        # BB
        for col, color in [("bb_upper", "#e53935"), ("bb_lower", "#43a047")]:
            if col in df_plot.columns:
                apds.append(mpf.make_addplot(df_plot[col], color=color, width=0.5, linestyle="--"))

        s = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size": 8})
        fig, axes = mpf.plot(
            df_plot, type="candle", style=s, addplot=apds, volume=True,
            figsize=(18, 12), title="",
            returnfig=True, warn_too_much_data=n+1,
        )
        ax_main = axes[0]
        ax_main.set_title(f"{ticker} 多策略信号图", fontproperties=_cn_font, fontsize=14, fontweight="bold")
        ax_vol = axes[2]

        # 标注每个策略的买入/卖出信号 (每种策略最多显示15个最新信号，避免过密)
        colors = ["#e91e63", "#2196f3", "#4caf50", "#ff9800", "#9c27b0", "#00bcd4", "#795548", "#e53935"]
        y_offset = 0.012 * (df["Close"].max() - df["Close"].min())
        for i, (sid, sig) in enumerate(signals.items()):
            name = STRATEGIES[sid].name if sid in STRATEGIES else sid
            color = colors[i % len(colors)]
            buy_idx = np.where(sig.values > 0)[0]
            sell_idx = np.where(sig.values < 0)[0]
            # 只显示最近的 15 个信号
            buy_idx = buy_idx[-15:] if len(buy_idx) > 15 else buy_idx
            sell_idx = sell_idx[-15:] if len(sell_idx) > 15 else sell_idx
            offset = y_offset * (i + 1)
            if len(buy_idx) > 0:
                ax_main.scatter(buy_idx, df["Close"].values[buy_idx] + offset,
                               marker="^", color=color, s=50, alpha=0.9,
                               label=f"{name} BUY")
            if len(sell_idx) > 0:
                ax_main.scatter(sell_idx, df["Close"].values[sell_idx] - offset,
                               marker="v", color=color, s=50, alpha=0.9,
                               label=f"{name} SELL")

        # 图例
        from matplotlib.lines import Line2D
        legend_items = []
        for i, (sid, _) in enumerate(signals.items()):
            name = STRATEGIES[sid].name if sid in STRATEGIES else sid
            color = colors[i % len(colors)]
            legend_items.append(Line2D([0], [0], marker="^", color="w", markerfacecolor=color, markersize=8, label=f"{name} 买入"))
            legend_items.append(Line2D([0], [0], marker="v", color="w", markerfacecolor=color, markersize=8, label=f"{name} 卖出"))
        if legend_items:
            ax_main.legend(handles=legend_items, loc="upper left", fontsize=7, ncol=2, framealpha=0.9, prop=_cn_font)

        # 保存
        out_dir = ROOT / "outputs" / "charts"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"{ticker}_strategy_signals.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return jsonify({
            "ticker": ticker.upper(),
            "chart_path": path,
            "strategies_shown": selected,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── 原有接口 ──

@app.route("/api/pattern_search/<ticker>")
def api_pattern_search(ticker):
    period = request.args.get("period", "1y")
    window = int(request.args.get("window", 30))
    top_n = int(request.args.get("top_n", 5))
    search_type = request.args.get("type", "historical")
    try:
        from src.pattern_search import (
            search_historical_similar, search_cross_stock_similar, get_pattern_features,
        )
        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)
        result = {"ticker": ticker.upper(), "type": search_type}
        if search_type == "historical":
            similar = search_historical_similar(df, window=window, top_n=top_n)
            result["similar_periods"] = similar
            result["pattern_features"] = get_pattern_features(df, window=window)
        elif search_type == "cross_stock":
            candidates = []
            candidate_names = []
            for other in ["AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN"]:
                if other == ticker.upper():
                    continue
                try:
                    other_df = fetch_data(other, period=period)
                    candidates.append(other_df)
                    candidate_names.append(other)
                except Exception:
                    pass
            similar = search_cross_stock_similar(df, candidates, candidate_names, window=window, top_n=top_n)
            result["similar_stocks"] = similar
            result["pattern_features"] = get_pattern_features(df, window=window)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/factors/<ticker>")
def api_factors(ticker):
    period = request.args.get("period", "1y")
    try:
        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)
        latest = df.iloc[-1]
        from src.factors import FACTOR_META
        factors = {}
        for key, (category, desc, _, _) in FACTOR_META.items():
            if key in latest.index:
                val = latest[key]
                if pd.isna(val):
                    continue
                factors[key] = {
                    "category": category,
                    "description": desc,
                    "value": round(float(val), 4) if not pd.isna(val) else None,
                }
        return jsonify({
            "ticker": ticker.upper(),
            "date": str(df["Date"].iloc[-1]),
            "factors": factors,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting FLOW AI Trading Dashboard...")
    print("  Modes: single / multi")
    print("  New: /api/strategies /api/backtest /api/consensus")
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
