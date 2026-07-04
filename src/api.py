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

# 缓存股票列表
_TICKERS_CACHE = None
_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ITEMS = 64
_DATA_CACHE = {}
_FACTOR_CACHE = {}
_BACKTEST_CACHE = {}
_PORTFOLIO_CACHE = {}
_DASHBOARD_CACHE = {}
_CACHE_STATS = {
    "data_hits": 0,
    "data_misses": 0,
    "factor_hits": 0,
    "factor_misses": 0,
    "backtest_hits": 0,
    "backtest_misses": 0,
    "portfolio_hits": 0,
    "portfolio_misses": 0,
    "dashboard_hits": 0,
    "dashboard_misses": 0,
}

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


def _cache_get(store, key, stat_hit=None, stat_miss=None):
    entry = store.get(key)
    now = time.time()
    if entry and now - entry["ts"] <= _CACHE_TTL_SECONDS:
        if stat_hit:
            _CACHE_STATS[stat_hit] += 1
        return entry["value"], True
    if entry:
        store.pop(key, None)
    if stat_miss:
        _CACHE_STATS[stat_miss] += 1
    return None, False


def _cache_put(store, key, value):
    if len(store) >= _CACHE_MAX_ITEMS:
        oldest = min(store, key=lambda k: store[k]["ts"])
        store.pop(oldest, None)
    store[key] = {"ts": time.time(), "value": value}


def _cached_fetch_data(ticker: str, period: str, interval: str = "1d"):
    key = (ticker.strip().upper(), period, interval)
    cached, hit = _cache_get(_DATA_CACHE, key, "data_hits", "data_misses")
    if hit:
        return cached.copy(), {"data": "hit"}
    df = fetch_data(ticker, period=period, interval=interval)
    _cache_put(_DATA_CACHE, key, df.copy())
    return df.copy(), {"data": "miss"}


def _factor_data(ticker: str, period: str):
    key = (ticker.strip().upper(), period)
    cached, hit = _cache_get(_FACTOR_CACHE, key, "factor_hits", "factor_misses")
    if hit:
        return cached.copy(), {"factor": "hit", "data": "skip"}
    df, cache_info = _cached_fetch_data(ticker, period)
    df = add_all_factors(df)
    _cache_put(_FACTOR_CACHE, key, df.copy())
    cache_info["factor"] = "miss"
    return df.copy(), cache_info


def _cached_backtest_all(df, ticker: str, period: str, capital: float,
                         worst_case: bool, limit_pct: float, hold_profile: str,
                         min_trades: int = 1):
    from src.strategies import backtest_all

    key = (
        ticker.strip().upper(), period, round(float(capital), 2),
        bool(worst_case), round(float(limit_pct), 4), hold_profile, int(min_trades),
    )
    cached, hit = _cache_get(_BACKTEST_CACHE, key, "backtest_hits", "backtest_misses")
    if hit:
        return cached, {"backtest": "hit"}
    results = backtest_all(
        df, initial_capital=capital, min_trades=min_trades,
        worst_case=worst_case, limit_pct=limit_pct, hold_profile=hold_profile,
    )
    _cache_put(_BACKTEST_CACHE, key, results)
    return results, {"backtest": "miss"}


def _with_meta(payload: dict, started_at: float, cache_info: dict | None = None):
    meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
    meta["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 1)
    if cache_info:
        meta["cache"] = cache_info
    meta["cache_stats"] = dict(_CACHE_STATS)
    payload["meta"] = meta
    return jsonify(payload)


def _strategy_payload(r, strategy=None):
    from src.strategies import (
        STRATEGIES, strategy_quality_score, strategy_grade,
        strategy_risk_level, strategy_recommendation_reason,
    )

    strat = strategy or STRATEGIES.get(r.strategy_id)
    score = strategy_quality_score(r)
    return {
        "name": r.strategy_name,
        "category": strat.category if strat else "",
        "description": strat.description if strat else "",
        "quality_score": score,
        "grade": strategy_grade(score),
        "risk_level": strategy_risk_level(r),
        "recommendation_reason": strategy_recommendation_reason(r),
        "total_return": f"{r.total_return * 100:.2f}%",
        "annual_return": f"{r.annual_return * 100:.2f}%",
        "sharpe_ratio": f"{r.sharpe_ratio:.2f}",
        "max_drawdown": f"{r.max_drawdown * 100:.2f}%",
        "win_rate": f"{r.win_rate * 100:.1f}%",
        "total_trades": r.total_trades,
        "avg_return_per_trade": f"{r.avg_return_per_trade * 100:.2f}%",
        "avg_hold_days": f"{r.avg_hold_days:.0f}天",
        "profit_factor": f"{r.profit_factor:.2f}",
    }


def _strategy_markers(df, ranked, max_strategies: int = 2, max_markers: int = 24):
    markers = []
    colors = ["#26a69a", "#2962ff", "#f5c542"]
    for rank, (sid, r, _) in enumerate(ranked[:max_strategies]):
        series = r.signal_series.reindex(df.index).fillna(0)
        idxs = np.where(series.values != 0)[0]
        if len(idxs) > 12:
            idxs = idxs[-12:]
        for idx in idxs:
            sig = int(series.iloc[idx])
            is_buy = sig > 0
            markers.append({
                "time": str(df["Date"].iloc[idx])[:10],
                "position": "belowBar" if is_buy else "aboveBar",
                "color": colors[rank % len(colors)] if is_buy else "#ef5350",
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": "买" if is_buy else "卖",
                "strategy_id": sid,
                "strategy_name": r.strategy_name,
                "signal": sig,
            })
    markers.sort(key=lambda item: item["time"])
    return markers[-max_markers:]


def _latest_factors_payload(df, ticker: str):
    from src.factors import FACTOR_META

    latest = df.iloc[-1]
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
    return {
        "ticker": ticker.upper(),
        "date": str(df["Date"].iloc[-1]),
        "factors": factors,
    }


def _build_portfolio_payload(df, ticker: str, period: str, hold_profile: str,
                             worst_case: bool, limit_pct: float, include_chart: bool,
                             bt_all: dict):
    from src.strategy_portfolio import analyze_portfolio, plot_portfolio_equity, backtest_portfolio
    from src.strategies import rank_strategies, HOLD_PROFILES

    summary = analyze_portfolio(
        df, worst_case=worst_case, limit_pct=limit_pct,
        hold_profile=hold_profile, precomputed_results=bt_all,
    )

    hp = HOLD_PROFILES[hold_profile]
    ranked = rank_strategies(bt_all, "quality_score")
    top5 = [sid for sid, _, _ in ranked[:5]]
    portfolio_result = backtest_portfolio(
        df, top5, worst_case=worst_case, limit_pct=limit_pct,
        stop_loss_pct=hp["sl"], take_profit_pct=hp["tp"],
        hold_days_min=hp["hold_min"], hold_days_max=hp["hold_max"],
    )
    chart_path = plot_portfolio_equity(df, portfolio_result, ticker, "等权Top5组合") if include_chart else None

    close = df["Close"].astype(float).values
    portfolio_values = portfolio_result.equity_curve.astype(float).values
    if len(close) and len(portfolio_values):
        buy_hold_equity = close / close[0]
        portfolio_equity = portfolio_values / portfolio_values[0]
        running_peak = np.maximum.accumulate(portfolio_equity)
        drawdown = (portfolio_equity - running_peak) / running_peak
        equity_curve = [
            {
                "date": str(df["Date"].iloc[i])[:10],
                "portfolio": round(float(portfolio_equity[i]), 5),
                "buy_hold": round(float(buy_hold_equity[i]), 5),
                "drawdown": round(float(drawdown[i] * 100), 2),
            }
            for i in range(min(len(df), len(portfolio_equity), len(buy_hold_equity)))
        ]
    else:
        equity_curve = []

    return {
        "ticker": ticker.upper(),
        "period": period,
        "hold_profile": hold_profile,
        "hold_label": HOLD_PROFILES[hold_profile]["label"],
        "hold_params": {
            "hold_min": hp["hold_min"], "hold_max": hp["hold_max"],
            "stop_loss_pct": round(hp["sl"] * 100, 1), "take_profit_pct": round(hp["tp"] * 100, 1),
        },
        "market_regime": summary.get("regime_info", {}),
        "portfolios": {k: v for k, v in summary.items() if k not in ["buy_hold", "top_strategies", "regime_info"]},
        "buy_hold": summary.get("buy_hold", {}),
        "top_strategies": summary.get("top_strategies", []),
        "portfolio_chart_path": chart_path,
        "equity_curve": equity_curve,
        "selected_strategies": top5,
    }


def _cached_portfolio_payload(df, ticker: str, period: str, hold_profile: str,
                              worst_case: bool, limit_pct: float, include_chart: bool,
                              bt_all: dict):
    key = (
        ticker.strip().upper(), period, hold_profile, bool(worst_case),
        round(float(limit_pct), 4), bool(include_chart),
    )
    cached, hit = _cache_get(_PORTFOLIO_CACHE, key, "portfolio_hits", "portfolio_misses")
    if hit:
        return cached, {"portfolio": "hit"}
    payload = _build_portfolio_payload(
        df, ticker, period, hold_profile, worst_case, limit_pct, include_chart, bt_all,
    )
    _cache_put(_PORTFOLIO_CACHE, key, payload)
    return payload, {"portfolio": "miss"}


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
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    mode = request.args.get("mode", "multi")
    include_chart = request.args.get("chart", "false").lower() in ("true", "1", "yes")
    try:
        df, cache_info = _factor_data(ticker, period)
        analysis = trend_analysis(df)
        df_sig = detect_all(df)
        sig = signal_summary(df_sig)
        active = get_active_signals(df_sig, recent_days=5)
        chart_path = plot_kline_trend(df, ticker, analysis, sig, mode=mode) if include_chart else None

        return _with_meta({
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
        }, started_at, cache_info)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/signals/<ticker>")
def api_signals(ticker):
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    try:
        df, cache_info = _factor_data(ticker, period)
        df = detect_all(df)
        sig = signal_summary(df)
        return _with_meta({"ticker": ticker.upper(), "period": period, "signals": sig}, started_at, cache_info)
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
      strategy:   指定策略ID (可选, 不传则回测全部)
      period:     数据周期 (默认1y)
      capital:    初始资金 (默认100000)
      worst_case: 最差执行模式 (true/false, 默认true)
      limit:      涨跌停限制百分比 (0=关闭, 默认0, 例如 limit=0.1 表示单日涨跌超10%禁止交易)
    """
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    strategy_id = request.args.get("strategy", None)
    capital = float(request.args.get("capital", 100000))
    worst_case = request.args.get("worst_case", "true").lower() in ("true", "1", "yes")
    limit_pct = float(request.args.get("limit", 0))
    hold_profile = request.args.get("hold", "medium")
    if hold_profile not in ("short", "medium", "long"):
        hold_profile = "medium"

    try:
        from src.strategies import (
            STRATEGIES, backtest_strategy, get_strategy_summary, HOLD_PROFILES,
            rank_strategies, strategy_quality_score, strategy_grade,
            strategy_risk_level, strategy_recommendation_reason,
        )

        df, cache_info = _factor_data(ticker, period)

        if strategy_id:
            if strategy_id not in STRATEGIES:
                return jsonify({"error": f"未知策略: {strategy_id}"}), 400
            r = backtest_strategy(df, strategy_id, initial_capital=capital,
                                  worst_case=worst_case, limit_pct=limit_pct,
                                  hold_override=HOLD_PROFILES[hold_profile])
            score = strategy_quality_score(r)
            result = {
                "ticker": ticker.upper(),
                "period": period,
                "strategy_id": strategy_id,
                "strategy_name": r.strategy_name,
                "quality_score": score,
                "grade": strategy_grade(score),
                "risk_level": strategy_risk_level(r),
                "recommendation_reason": strategy_recommendation_reason(r),
                "total_return": round(r.total_return * 100, 2),
                "annual_return": round(r.annual_return * 100, 2),
                "sharpe_ratio": round(r.sharpe_ratio, 2),
                "max_drawdown": round(r.max_drawdown * 100, 2),
                "win_rate": round(r.win_rate * 100, 1),
                "total_trades": r.total_trades,
                "avg_return_per_trade": round(r.avg_return_per_trade * 100, 2),
                "avg_hold_days": round(r.avg_hold_days, 0),
                "profit_factor": round(r.profit_factor, 2),
                "worst_case": worst_case,
                "limit_pct": limit_pct,
                "hold_profile": hold_profile,
                # 最近20天的信号
                "recent_signals": r.signal_series.iloc[-20:].tolist(),
            }
        else:
            results, bt_cache = _cached_backtest_all(
                df, ticker, period, capital, worst_case, limit_pct, hold_profile,
                min_trades=1,
            )
            cache_info.update(bt_cache)
            summary_df = get_strategy_summary(results)
            ranked_by_score = rank_strategies(results, "quality_score")
            ranked_by_sharpe = rank_strategies(results, "sharpe_ratio")
            result = {
                "ticker": ticker.upper(),
                "period": period,
                "hold_profile": hold_profile,
                "hold_label": HOLD_PROFILES[hold_profile]["label"],
                "worst_case": worst_case,
                "strategies": {},
                "ranked_by_score": [],
                "ranked_by_sharpe": [],
                "signal_markers": _strategy_markers(df, ranked_by_score),
                "recommendation": {},
            }
            for _, row in summary_df.iterrows():
                sid = row["策略ID"]
                result["strategies"][sid] = _strategy_payload(results[sid], STRATEGIES.get(sid))
            result["ranked_by_score"] = [sid for sid, _, _ in ranked_by_score]
            result["ranked_by_sharpe"] = [sid for sid, _, _ in ranked_by_sharpe]
            if ranked_by_score:
                top_sid, top_result, top_score = ranked_by_score[0]
                result["recommendation"] = {
                    "primary_strategy_id": top_sid,
                    "primary_strategy": top_result.strategy_name,
                    "score": int(top_score),
                    "grade": strategy_grade(int(top_score)),
                    "risk_level": strategy_risk_level(top_result),
                    "reason": strategy_recommendation_reason(top_result),
                    "rank_basis": "综合评分综合收益、夏普、最大回撤、胜率和交易样本数",
                }

        return _with_meta(result, started_at, cache_info)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/consensus/<ticker>")
def api_consensus(ticker):
    """
    获取多策略共识信号
    返回所有策略的最新信号 + 综合评分
    """
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    try:
        from src.strategies import get_latest_consensus, aggregate_signals

        df, cache_info = _factor_data(ticker, period)

        consensus = get_latest_consensus(df)

        # 额外：获取最近20天的共识分数趋势
        agg = aggregate_signals(df)
        recent = agg.iloc[-60:][["aggregate_score", "bullish_ratio"]].copy()
        recent["date"] = df["Date"].iloc[-60:].values

        return _with_meta({
            "ticker": ticker.upper(),
            "consensus": consensus,
            "trend": [
                {"date": str(row["date"])[:10], "score": float(row["aggregate_score"]), "ratio": float(row["bullish_ratio"])}
                for _, row in recent.iterrows()
            ],
        }, started_at, cache_info)
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
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    worst_case = request.args.get("worst_case", "true").lower() in ("true", "1", "yes")
    limit_pct = float(request.args.get("limit", 0))
    hold_profile = request.args.get("hold", "medium")
    if hold_profile not in ("short", "medium", "long"):
        hold_profile = "medium"
    include_chart = request.args.get("chart", "false").lower() in ("true", "1", "yes")

    try:
        df, cache_info = _factor_data(ticker, period)
        bt_all, bt_cache = _cached_backtest_all(
            df, ticker, period, 100000, worst_case, limit_pct, hold_profile,
            min_trades=1,
        )
        cache_info.update(bt_cache)
        payload, portfolio_cache = _cached_portfolio_payload(
            df, ticker, period, hold_profile, worst_case, limit_pct, include_chart, bt_all,
        )
        cache_info.update(portfolio_cache)
        return _with_meta(payload.copy(), started_at, cache_info)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/dashboard/<ticker>")
def api_dashboard(ticker):
    """主页面聚合接口：一次请求复用行情、因子、回测和组合中间结果。"""
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    mode = request.args.get("mode", "multi")
    capital = float(request.args.get("capital", 100000))
    worst_case = request.args.get("worst_case", "true").lower() in ("true", "1", "yes")
    limit_pct = float(request.args.get("limit", 0))
    hold_profile = request.args.get("hold", "medium")
    if hold_profile not in ("short", "medium", "long"):
        hold_profile = "medium"
    include_chart = request.args.get("chart", "false").lower() in ("true", "1", "yes")
    dashboard_key = (
        ticker.strip().upper(), period, mode, round(float(capital), 2),
        bool(worst_case), round(float(limit_pct), 4), hold_profile, bool(include_chart),
    )
    cached_dashboard, dashboard_hit = _cache_get(
        _DASHBOARD_CACHE, dashboard_key, "dashboard_hits", "dashboard_misses",
    )
    if dashboard_hit:
        payload = cached_dashboard.copy()
        return _with_meta(payload, started_at, {"dashboard": "hit"})

    try:
        from src.strategies import (
            get_strategy_summary, rank_strategies, strategy_grade,
            strategy_risk_level, strategy_recommendation_reason,
            get_latest_consensus, aggregate_signals, HOLD_PROFILES, STRATEGIES,
        )

        df, cache_info = _factor_data(ticker, period)

        analysis = trend_analysis(df)
        df_sig = detect_all(df)
        sig = signal_summary(df_sig)
        active = get_active_signals(df_sig, recent_days=5)
        chart_path = plot_kline_trend(df, ticker, analysis, sig, mode=mode) if include_chart else None
        analyze_payload = {
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
        }

        results, bt_cache = _cached_backtest_all(
            df, ticker, period, capital, worst_case, limit_pct, hold_profile,
            min_trades=1,
        )
        cache_info.update(bt_cache)
        summary_df = get_strategy_summary(results)
        ranked_by_score = rank_strategies(results, "quality_score")
        ranked_by_sharpe = rank_strategies(results, "sharpe_ratio")
        backtest_payload = {
            "ticker": ticker.upper(),
            "period": period,
            "hold_profile": hold_profile,
            "hold_label": HOLD_PROFILES[hold_profile]["label"],
            "worst_case": worst_case,
            "strategies": {},
            "ranked_by_score": [sid for sid, _, _ in ranked_by_score],
            "ranked_by_sharpe": [sid for sid, _, _ in ranked_by_sharpe],
            "signal_markers": _strategy_markers(df, ranked_by_score),
            "recommendation": {},
        }
        for _, row in summary_df.iterrows():
            sid = row["策略ID"]
            backtest_payload["strategies"][sid] = _strategy_payload(results[sid], STRATEGIES.get(sid))
        if ranked_by_score:
            top_sid, top_result, top_score = ranked_by_score[0]
            backtest_payload["recommendation"] = {
                "primary_strategy_id": top_sid,
                "primary_strategy": top_result.strategy_name,
                "score": int(top_score),
                "grade": strategy_grade(int(top_score)),
                "risk_level": strategy_risk_level(top_result),
                "reason": strategy_recommendation_reason(top_result),
                "rank_basis": "综合评分综合收益、夏普、最大回撤、胜率和交易样本数",
            }

        consensus = get_latest_consensus(df)
        agg = aggregate_signals(df)
        recent = agg.iloc[-60:][["aggregate_score", "bullish_ratio"]].copy()
        recent["date"] = df["Date"].iloc[-60:].values
        consensus_payload = {
            "ticker": ticker.upper(),
            "consensus": consensus,
            "trend": [
                {"date": str(row["date"])[:10], "score": float(row["aggregate_score"]), "ratio": float(row["bullish_ratio"])}
                for _, row in recent.iterrows()
            ],
        }

        portfolio_payload, portfolio_cache = _cached_portfolio_payload(
            df, ticker, period, hold_profile, worst_case, limit_pct, include_chart, results,
        )
        cache_info.update(portfolio_cache)

        payload = {
            "ticker": ticker.upper(),
            "period": period,
            "analyze": analyze_payload,
            "backtest": backtest_payload,
            "consensus": consensus_payload,
            "portfolio": portfolio_payload,
            "factors": _latest_factors_payload(df, ticker),
        }
        _cache_put(_DASHBOARD_CACHE, dashboard_key, payload.copy())
        cache_info["dashboard"] = "miss"
        return _with_meta(payload, started_at, cache_info)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/strategy_chart/<ticker>")
def api_strategy_chart(ticker):
    """
    生成策略组合分析图
    包含: K线 + 多策略信号标注 + 权益曲线
    """
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    strategy_id = request.args.get("strategy", None)

    try:
        from src.strategies import STRATEGIES, backtest_strategy, aggregate_signals
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mplfinance as mpf
        import numpy as np
        import pandas as pd
        from src.plot_fonts import configure_chinese_font

        _cn_font = configure_chinese_font()

        df, cache_info = _factor_data(ticker, period)

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

        return _with_meta({
            "ticker": ticker.upper(),
            "chart_path": path,
            "strategies_shown": selected,
        }, started_at, cache_info)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── 原有接口 ──

@app.route("/api/pattern_search/<ticker>")
def api_pattern_search(ticker):
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    window = int(request.args.get("window", 30))
    top_n = int(request.args.get("top_n", 5))
    search_type = request.args.get("type", "historical")
    try:
        from src.pattern_search import (
            search_historical_similar, search_cross_stock_similar, get_pattern_features,
        )
        df, cache_info = _factor_data(ticker, period)
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
                    other_df, _ = _cached_fetch_data(other, period)
                    candidates.append(other_df)
                    candidate_names.append(other)
                except Exception:
                    pass
            similar = search_cross_stock_similar(df, candidates, candidate_names, window=window, top_n=top_n)
            result["similar_stocks"] = similar
            result["pattern_features"] = get_pattern_features(df, window=window)
        return _with_meta(result, started_at, cache_info)
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
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    limit = request.args.get("limit", None)
    fields = request.args.get("fields", None)

    try:
        df, cache_info = _factor_data(ticker, period)

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

        return _with_meta({
            "ticker": ticker.upper(),
            "period": period,
            "total_records": len(df),
            "returned_records": len(records),
            "available_fields": [c for c in df.columns if c != "Date"],
            "data": records,
        }, started_at, cache_info)
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
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    date = request.args.get("date", None)

    try:
        from src.strategies import get_latest_consensus, STRATEGIES, backtest_all, rank_strategies
        from src.factors import FACTOR_META

        df, cache_info = _factor_data(ticker, period)
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

        return _with_meta({
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
        }, started_at, cache_info)
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
    started_at = time.perf_counter()
    period = request.args.get("period", "1y")
    try:
        df, cache_info = _factor_data(ticker, period)
        return _with_meta(_latest_factors_payload(df, ticker), started_at, cache_info)
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
