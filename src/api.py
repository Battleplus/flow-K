"""Flask API + 仪表盘服务端 — 集成因子/多面板/形态搜索/策略引擎"""

from flask import Flask, jsonify, request, send_from_directory, send_file
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_data
from src.indicators import trend_analysis
from src.patterns import add_indicators, detect_all, signal_summary, get_active_signals
from src.chart import plot_kline_trend
from src.factors import add_all_factors
from src.strategy_portfolio import analyze_portfolio, plot_portfolio_equity

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


@app.route("/api/portfolio/<ticker>")
def api_portfolio(ticker):
    """
    策略组合分析
    返回:
      - 市场状态识别
      - 多种组合方式回测结果 (等权/夏普加权/状态驱动/动态波动率)
      - 买入持有基准
      - 组合权益曲线图
    """
    period = request.args.get("period", "1y")
    try:
        from src.strategy_portfolio import analyze_portfolio, plot_portfolio_equity, backtest_portfolio
        from src.strategies import backtest_all, rank_strategies

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)

        summary = analyze_portfolio(df)

        # 用等权 Top5 组合生成权益曲线图
        ranked = rank_strategies(backtest_all(df), "sharpe_ratio")
        top5 = [sid for sid, _, _ in ranked[:5]]
        portfolio_result = backtest_portfolio(df, top5)
        chart_path = plot_portfolio_equity(df, portfolio_result, ticker, "等权Top5组合")

        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "market_regime": summary.get("regime_info", {}),
            "portfolios": {k: v for k, v in summary.items() if k not in ["buy_hold", "top_strategies", "regime_info"]},
            "buy_hold": summary.get("buy_hold", {}),
            "top_strategies": summary.get("top_strategies", []),
            "portfolio_chart_path": chart_path,
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


# ═══════════════════════════════════════════════════════════════
# 天级数据 API
# ═══════════════════════════════════════════════════════════════

@app.route("/api/daily/<ticker>")
def api_daily(ticker):
    """
    获取天级 K 线数据 (JSON 格式)
    参数:
      period: 时间范围 (1y/6mo/3mo/1mo/5y/max)
      limit:  返回最近 N 条 (可选，默认全部)
      fields: 返回字段，逗号分隔 (open,high,low,close,volume,ma5,ma10...)
              不传则返回全部可用字段
    """
    period = request.args.get("period", "1y")
    limit = request.args.get("limit", None)
    fields = request.args.get("fields", None)

    try:
        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)

        if limit:
            limit = int(limit)
            df = df.iloc[-limit:]

        # 字段过滤
        if fields:
            wanted = [f.strip() for f in fields.split(",")]
            available = [f for f in wanted if f in df.columns]
            df = df[["Date"] + available] if available else df

        # 构建 JSON
        records = []
        for _, row in df.iterrows():
            rec = {"date": str(row["Date"])[:10]}
            for col in df.columns:
                if col == "Date":
                    continue
                val = row[col]
                if pd.isna(val):
                    rec[col] = None
                elif isinstance(val, (float, np.floating)):
                    rec[col] = round(float(val), 4)
                elif isinstance(val, (int, np.integer)):
                    rec[col] = int(val)
                else:
                    rec[col] = val
            records.append(rec)

        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "total_records": len(df),
            "returned_records": len(records),
            "available_fields": [c for c in df.columns if c != "Date"],
            "data": records,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/daily_summary/<ticker>")
def api_daily_summary(ticker):
    """
    单日综合摘要 — 因子值 + 信号 + 策略共识
    参数:
      period: 数据周期 (默认1y，用于因子计算历史)
      date:   指定日期 (可选，默认最新交易日)
    """
    period = request.args.get("period", "1y")
    date = request.args.get("date", None)

    try:
        from src.strategies import get_latest_consensus, STRATEGIES, backtest_all, rank_strategies
        from src.factors import FACTOR_META

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)
        df_sig = detect_all(df)

        # 指定日期或最新
        if date:
            target = pd.Timestamp(date).date()
            matches = df[df["Date"] == target]
            if len(matches) == 0:
                return jsonify({"error": f"日期 {date} 不在数据范围内"}), 400
            row_idx = len(df) - len(df[df["Date"] > target]) - 1
            target_date = target
        else:
            row_idx = len(df) - 1
            target_date = df["Date"].iloc[-1]

        row = df.iloc[row_idx]

        # 因子值
        factors = {}
        for key, (category, desc, _, _) in FACTOR_META.items():
            if key in row.index and not pd.isna(row[key]):
                factors[key] = {
                    "category": category,
                    "description": desc,
                    "value": round(float(row[key]), 4),
                }

        # 信号
        sig = signal_summary(df_sig)
        active = get_active_signals(df_sig, recent_days=5)

        # 策略共识
        consensus = get_latest_consensus(df)

        # 当日 K 线
        ohlcv = {
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        }

        # 涨跌
        if row_idx > 0:
            prev_close = float(df["Close"].iloc[row_idx - 1])
            chg = ohlcv["close"] - prev_close
            chg_pct = (chg / prev_close) * 100
        else:
            chg = 0
            chg_pct = 0

        return jsonify({
            "ticker": ticker.upper(),
            "date": str(target_date)[:10],
            "ohlcv": ohlcv,
            "change": round(chg, 2),
            "change_pct": round(chg_pct, 2),
            "factor_count": len(factors),
            "factors": factors,
            "signals": {
                "score": sig["score"],
                "verdict": sig["verdict"],
                "bullish_count": sig["bullish_count"],
                "bearish_count": sig["bearish_count"],
                "neutral_count": sig["neutral_count"],
                "active": active,
            },
            "strategy_consensus": consensus,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/market_snapshot")
def api_market_snapshot():
    """大盘快照 — 主要指数 + 板块 ETF 行情"""
    try:
        from src.data_loader import fetch_market_indices
        snap = fetch_market_indices()
        return jsonify(snap)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
# 日内实时数据 API
# ═══════════════════════════════════════════════════════════════

@app.route("/api/intraday/<ticker>")
def api_intraday(ticker):
    """
    获取日内分时数据
    参数:
      interval: K线粒度 (1m/2m/5m/15m/30m/60m/1h), 默认5m
      period:   时间范围 (1d/5d/1mo), 默认1d
    """
    interval = request.args.get("interval", "5m")
    period = request.args.get("period", "1d")

    try:
        from src.data_loader import fetch_intraday
        result = fetch_intraday(ticker, period=period, interval=interval)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/quote/<ticker>")
def api_quote(ticker):
    """获取实时报价快照"""
    try:
        from src.data_loader import fetch_quote
        result = fetch_quote(ticker)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/quotes")
def api_quotes_batch():
    """
    批量获取实时报价
    参数:
      tickers: 逗号分隔的股票代码 (如 AAPL,MSFT,GOOGL)
    """
    tickers_str = request.args.get("tickers", "")
    if not tickers_str:
        return jsonify({"error": "请提供 tickers 参数"}), 400

    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        return jsonify({"error": "无有效股票代码"}), 400
    if len(tickers) > 20:
        return jsonify({"error": "最多支持20只股票同时查询"}), 400

    try:
        from src.data_loader import fetch_quote
        results = {}
        for t in tickers:
            try:
                results[t] = fetch_quote(t)
            except Exception as err:
                results[t] = {"error": str(err)}
        return jsonify({
            "count": len(tickers),
            "quotes": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── 原有接口 ──

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
    print("=" * 60)
    print("  FLOW AI Trading Dashboard")
    print("=" * 60)
    print("  天级数据:   /api/daily/<ticker>")
    print("               /api/daily_summary/<ticker>")
    print("               /api/market_snapshot")
    print("  日内实时:   /api/quote/<ticker>")
    print("               /api/quotes?tickers=AAPL,MSFT")
    print("               /api/intraday/<ticker>?interval=5m")
    print("  策略/组合:  /api/backtest/<ticker>")
    print("               /api/consensus/<ticker>")
    print("               /api/portfolio/<ticker>")
    print("  分析/因子:  /api/analyze/<ticker>")
    print("               /api/factors/<ticker>")
    print("=" * 60)
    print("  Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
