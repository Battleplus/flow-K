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
from src.news_analysis import analyze_news
from src.sector_rotation import get_sector_snapshot
from src.three_dim_analyzer import three_dim_scan, three_dim_single

app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="")

# 缓存股票列表
_TICKERS_CACHE = None

def _load_tickers():
    global _TICKERS_CACHE
    if _TICKERS_CACHE is not None:
        return _TICKERS_CACHE
    result = []
    # 从 company_info.csv 读取
    info_path = ROOT / "data" / "company_info.csv"
    if info_path.exists():
        try:
            info = pd.read_csv(info_path)
            for _, r in info.iterrows():
                sym = str(r.get("Symbol","")).strip()
                if sym and sym != "nan":
                    result.append({
                        "symbol": sym,
                        "name": str(r.get("Company",""))[:30],
                        "sector": str(r.get("Sector",""))[:20],
                    })
        except Exception:
            pass
    # 从历史CSV读取有数据的股票
    csv_path = ROOT / "SP500_Historical_Data.csv"
    if csv_path.exists():
        try:
            hist_syms = set(pd.read_csv(csv_path, usecols=["Ticker"])["Ticker"].unique())
            # 只保留有历史数据的
            result = [t for t in result if t["symbol"] in hist_syms]
            # 补上CSV里有但info里没有的
            known = {t["symbol"] for t in result}
            for sym in hist_syms:
                if sym not in known:
                    result.append({"symbol": sym, "name": "", "sector": ""})
        except Exception:
            pass
    _TICKERS_CACHE = result
    return result


@app.route("/api/tickers")
def api_tickers():
    """返回所有可用股票代码+名称+行业，供下拉选择"""
    tickers = _load_tickers()
    q = request.args.get("q", "").upper().strip()
    if q:
        tickers = [t for t in tickers if q in t["symbol"] or q.upper() in t["name"].upper()]
    return jsonify({"count": len(tickers), "tickers": tickers[:200]})


@app.route("/")
def index():
    return send_from_directory(str(ROOT / "web"), "index.html")


@app.route("/report")
def report():
    return send_from_directory(str(ROOT / "outputs"), "backtest_report.html")


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
    Walk-Forward 回测（主）+ 传统样本内回测（参考）

    参数:
      strategy:   指定策略ID (可选, 不传则回测全部)
      period:     数据周期 (自动升级: short=1y, medium=2y, long=3y)
      capital:    初始资金 (默认100000)
      worst_case: 最差执行模式 (true/false, 默认true)
      hold:       持仓周期 short/medium/long
    """
    hold_profile = request.args.get("hold", "medium")
    if hold_profile not in ("short", "medium", "long"):
        hold_profile = "medium"

    # 自动升级数据周期，确保 WF 有足够数据
    auto_period = {"short": "1y", "medium": "2y", "long": "3y"}
    period = request.args.get("period", auto_period.get(hold_profile, "2y"))

    strategy_id = request.args.get("strategy", None)
    capital = float(request.args.get("capital", 100000))
    worst_case = request.args.get("worst_case", "true").lower() in ("true", "1", "yes")
    limit_pct = float(request.args.get("limit", 0))

    try:
        from src.strategies import STRATEGIES, backtest_strategy, backtest_all, get_strategy_summary, HOLD_PROFILES, rank_strategies
        from src.walkforward import walk_forward, result_to_dict as wf_to_dict

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)

        # === Walk-Forward 样本外回测（主结果）===
        wf = walk_forward(df, hold_profile=hold_profile, worst_case=worst_case,
                          initial_capital=capital, top_k=5)
        wf_dict = wf_to_dict(wf)

        # === 传统样本内回测（参考对比）===
        results = backtest_all(df, initial_capital=capital, min_trades=1,
                               worst_case=worst_case, limit_pct=limit_pct,
                               hold_profile=hold_profile)
        summary_df = get_strategy_summary(results)

        in_sample_strategies = {}
        for _, row in summary_df.iterrows():
            sid = row["策略ID"]
            in_sample_strategies[sid] = {
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
        ranked = rank_strategies(results, "sharpe_ratio")

        # 买入持有
        bh_return = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]

        result = {
            "ticker": ticker.upper(),
            "period": period,
            "hold_profile": hold_profile,
            "hold_label": HOLD_PROFILES[hold_profile]["label"],
            "worst_case": worst_case,
            "n_data_days": len(df),
            "data_start": str(df["Date"].iloc[0]) if "Date" in df.columns else str(df.index[0]),
            "data_end": str(df["Date"].iloc[-1]) if "Date" in df.columns else str(df.index[-1]),

            # 主结果：Walk-Forward 样本外
            "walk_forward": wf_dict,

            # 参考对比：传统样本内
            "in_sample_reference": {
                "strategies": in_sample_strategies,
                "ranked_by_sharpe": [sid for sid, _, _ in ranked],
            },

            # 基准
            "buy_hold_return": round(bh_return * 100, 2),
            "overfit_gap": round(wf_dict["in_sample_return"] - wf_dict["total_return"], 2),
        }

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
    Walk-Forward 组合回测（主）+ 传统样本内组合（参考）

    主结果: Walk-Forward 样本外组合表现
    参考:   传统全量样本内组合分析 + 市场状态
    """
    hold_profile = request.args.get("hold", "medium")
    if hold_profile not in ("short", "medium", "long"):
        hold_profile = "medium"

    auto_period = {"short": "1y", "medium": "2y", "long": "3y"}
    period = request.args.get("period", auto_period.get(hold_profile, "2y"))
    worst_case = request.args.get("worst_case", "true").lower() in ("true", "1", "yes")
    limit_pct = float(request.args.get("limit", 0))

    try:
        from src.strategy_portfolio import analyze_portfolio, detect_market_regime
        from src.strategies import HOLD_PROFILES
        from src.walkforward import walk_forward, result_to_dict as wf_to_dict

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)
        hp = HOLD_PROFILES[hold_profile]

        # === Walk-Forward 样本外组合（主结果）===
        wf = walk_forward(df, hold_profile=hold_profile, worst_case=worst_case,
                          initial_capital=100000, top_k=5)
        wf_dict = wf_to_dict(wf)

        # === 传统样本内组合（参考对比）===
        summary = analyze_portfolio(df, worst_case=worst_case, limit_pct=limit_pct,
                                    hold_profile=hold_profile)

        # 买入持有
        bh_return = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]

        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "hold_profile": hold_profile,
            "hold_label": hp["label"],
            "hold_params": {
                "hold_min": hp["hold_min"], "hold_max": hp["hold_max"],
                "stop_loss_pct": round(hp["sl"] * 100, 1), "take_profit_pct": round(hp["tp"] * 100, 1),
            },
            "n_data_days": len(df),
            "data_start": str(df["Date"].iloc[0]) if "Date" in df.columns else str(df.index[0]),
            "data_end": str(df["Date"].iloc[-1]) if "Date" in df.columns else str(df.index[-1]),

            # 主结果
            "walk_forward": wf_dict,

            # 参考对比
            "in_sample_reference": {
                "portfolios": {k: v for k, v in summary.items() if k not in ["buy_hold", "top_strategies", "regime_info"]},
                "top_strategies": summary.get("top_strategies", []),
            },
            "market_regime": summary.get("regime_info", {}),

            # 基准
            "buy_hold_return": round(bh_return * 100, 2),
            "overfit_gap": round(wf_dict["in_sample_return"] - wf_dict["total_return"], 2),
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


# ── 三维分析 ──────────────────────────────────────────────
@app.route("/api/news_analyze", methods=["POST"])
def news_analyze():
    """消息面分析：输入新闻文本，返回板块评分"""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "请提供新闻文本 (text)"}), 400
    try:
        result = analyze_news(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sector_snapshot")
def sector_snapshot():
    """板块ETF快照：11个板块的趋势+RS+资金流向"""
    try:
        snap = get_sector_snapshot()
        return jsonify(snap)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/three_dim_scan", methods=["POST"])
def three_dim_scan_api():
    """三维全链路扫描：新闻→板块→个股→TOP推荐"""
    data = request.get_json(silent=True) or {}
    news_text = data.get("news_text", "").strip() or None
    period = data.get("period", "6mo")
    top_n = min(int(data.get("top_n", 10)), 20)  # 上限20
    try:
        result = three_dim_scan(news_text=news_text, period=period, top_n=top_n)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/three_dim/<ticker>")
def three_dim_ticker(ticker):
    """单只股票三维分析详情"""
    period = request.args.get("period", "6mo")
    try:
        result = three_dim_single(ticker.upper(), period=period)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommendations")
def api_recommendations():
    """三维推荐结果：返回预扫描的JSON数据"""
    import json as _json
    rec_path = ROOT / "outputs" / "stock_recommendations.json"
    if not rec_path.exists():
        return jsonify({"error": "推荐数据尚未生成，请先运行 src/full_scan.py"}), 404
    try:
        with open(rec_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommendations/run", methods=["POST"])
def api_recommendations_run():
    """触发三维全板块扫描（异步，返回结果到 /api/recommendations）"""
    import json as _json
    data = request.get_json(silent=True) or {}
    news_text = data.get("news_text", "").strip() or "AI semiconductor technology data center cloud computing chips GPU"
    top_n = min(int(data.get("top_n", 10)), 20)

    try:
        from src.screener import analyze_ticker
        from src.news_analysis import analyze_news
        from src.sector_rotation import get_sector_snapshot

        TICKERS = {
            "Technology": ["AAPL","MSFT","NVDA","AVGO","AMD","GOOGL","META","ADBE","CRM","INTC"],
            "Financials": ["JPM","BAC","WFC","V","MA","GS","MS","BLK"],
            "Healthcare": ["JNJ","UNH","PFE","ABBV","LLY","MRK","TMO","ABT"],
            "Consumer Cyclical": ["AMZN","TSLA","HD","NKE","MCD","SBUX","LOW","F"],
            "Communication Services": ["GOOGL","META","NFLX","DIS","CMCSA","TMUS"],
            "Industrials": ["CAT","GE","HON","UPS","RTX","LMT","DE","EMR"],
            "Consumer Defensive": ["PG","KO","PEP","WMT","COST","MO","PM"],
            "Energy": ["XOM","CVX","COP","SLB","EOG","PSX"],
            "Utilities": ["NEE","DUK","SO","D","AEP","SRE"],
            "Real Estate": ["PLD","AMT","EQIX","SPG","O","PSA"],
            "Basic Materials": ["LIN","FCX","NEM","APD","DOW","ECL"],
        }

        news = analyze_news(news_text)
        snap = get_sector_snapshot()
        sector_flow = {}
        for s in snap["sectors"]:
            sector_flow[s["sector"]] = {
                "flow": s["flow_score"],
                "rs": s["rs_score"],
                "rs_short": s["rs_short_score"],
            }

        all_results = []
        for sector, tickers in TICKERS.items():
            ns = news["sector_scores"].get(sector, 0)
            sf = sector_flow.get(sector, {})
            sector_score = ns * 0.4 + sf.get("flow", 0) * 0.6

            for ticker in tickers:
                try:
                    info = analyze_ticker(ticker, period="6mo")
                    if info is None:
                        continue
                except Exception:
                    continue

                direction_map = {
                    "strong_bullish": 2.0, "bullish": 1.0,
                    "neutral": 0.0, "bearish": -1.0, "strong_bearish": -2.0,
                }
                base = direction_map.get(info["consensus_direction"], 0.0)
                adj = max(-0.5, min(0.5, info["aggregate_score"] * 0.2))
                stock_3d = round(max(-2.0, min(2.0, base + adj)), 1)

                final = round(ns * 0.25 + sector_score * 0.35 + stock_3d * 0.4, 1)
                final = max(-3.0, min(3.0, final))

                all_results.append({
                    "ticker": ticker,
                    "sector": sector,
                    "news": round(ns, 1),
                    "sector_sc": round(sector_score, 1),
                    "stock": stock_3d,
                    "final": final,
                    "direction": info["consensus_direction"],
                    "sharpe": round(info.get("avg_sharpe_top5", 0), 2),
                    "pf_return": round(info.get("pf_return", 0), 1),
                    "signal_net": info.get("signal_net", 0),
                    "close": info.get("close", 0),
                })

        all_results.sort(key=lambda x: x["final"], reverse=True)

        by_sector = {}
        for r in all_results:
            s = r["sector"]
            if s not in by_sector:
                by_sector[s] = []
            by_sector[s].append(r)
        by_sector = {k: sorted(v, key=lambda x: x["final"], reverse=True) for k, v in by_sector.items()}

        output = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "news_text": news_text,
            "sector_snapshot": snap["sectors"],
            "all_results": all_results,
            "by_sector": by_sector,
            "top_10": all_results[:top_n],
        }

        rec_path = ROOT / "outputs" / "stock_recommendations.json"
        with open(rec_path, "w", encoding="utf-8") as f:
            _json.dump(output, f, indent=2, default=str, ensure_ascii=False)

        return jsonify(output)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/walkforward/<ticker>")
def api_walkforward(ticker):
    """
    Walk-Forward 滚动验证回测

    参数:
      period:  数据周期 (默认2y)
      hold:    持仓周期 short/medium/long (默认medium)
      top_k:   每折选多少策略 (默认5)
      compare: true=同时返回传统回测对比 (默认true)
    """
    period = request.args.get("period", "2y")
    hold_profile = request.args.get("hold", "medium")
    if hold_profile not in ("short", "medium", "long"):
        hold_profile = "medium"
    top_k = int(request.args.get("top_k", 5))
    do_compare = request.args.get("compare", "true").lower() in ("true", "1", "yes")

    try:
        from src.walkforward import walk_forward, result_to_dict, compare_in_sample_vs_oos, WF_WINDOWS

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)

        if do_compare:
            result = compare_in_sample_vs_oos(df, hold_profile=hold_profile, worst_case=True, top_k=top_k)
            result["ticker"] = ticker.upper()
            result["hold_profile"] = hold_profile
            result["data_days"] = len(df)
            result["window_config"] = WF_WINDOWS.get(hold_profile, {})
            return jsonify(result)
        else:
            wf = walk_forward(df, hold_profile=hold_profile, worst_case=True, top_k=top_k)
            d = result_to_dict(wf)
            d["ticker"] = ticker.upper()
            d["hold_profile"] = hold_profile
            d["data_days"] = len(df)
            return jsonify(d)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  FLOW AI Trading Dashboard v2 — 三维分析")
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
    print("  Walk-Forward: /api/walkforward/<ticker>")
    print("  分析/因子:  /api/analyze/<ticker>")
    print("               /api/factors/<ticker>")
    print("  🆕 三维分析: /api/news_analyze (POST)")
    print("               /api/sector_snapshot")
    print("               /api/three_dim_scan (POST)")
    print("               /api/three_dim/<ticker>")
    print("               /api/recommendations")
    print("               /api/recommendations/run (POST)")
    print("=" * 60)
    print("  Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
