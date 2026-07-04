"""Flask API + 仪表盘服务端 — 集成因子/多面板/形态搜索"""

from flask import Flask, jsonify, request, send_from_directory, send_file
import sys
from pathlib import Path
import pandas as pd

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
    mode = request.args.get("mode", "multi")   # "single" or "multi"
    try:
        df = fetch_data(ticker, period=period)

        # 计算所有因子
        df = add_all_factors(df)

        # 基础趋势分析
        analysis = trend_analysis(df)

        # 信号系统分析
        df_sig = detect_all(df)
        sig = signal_summary(df_sig)
        active = get_active_signals(df_sig, recent_days=5)

        # 绘图 (支持 single/multi 模式)
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
    """单独获取信号详情"""
    period = request.args.get("period", "1y")
    try:
        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)
        df = detect_all(df)
        sig = signal_summary(df)
        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "signals": sig,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pattern_search/<ticker>")
def api_pattern_search(ticker):
    """
    形态相似搜索
    参数:
      window: 匹配窗口长度 (默认30天)
      top_n:  返回数量 (默认5)
      type:   "historical"=历史相似 / "cross_stock"=跨股票相似
    """
    period = request.args.get("period", "1y")
    window = int(request.args.get("window", 30))
    top_n = int(request.args.get("top_n", 5))
    search_type = request.args.get("type", "historical")

    try:
        from src.pattern_search import (
            search_historical_similar,
            search_cross_stock_similar,
            get_pattern_features,
        )

        df = fetch_data(ticker, period=period)
        df = add_all_factors(df)

        result = {"ticker": ticker.upper(), "type": search_type}

        if search_type == "historical":
            similar = search_historical_similar(df, window=window, top_n=top_n)
            result["similar_periods"] = similar
            result["pattern_features"] = get_pattern_features(df, window=window)
        elif search_type == "cross_stock":
            # 与其他SP500股票比较 (取data目录下的CSV或在线获取)
            # 简化: 与几个主流股票比较
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
    """返回所有因子的最新值 (用于前端表格展示)"""
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
    print("  Modes: single (单面板) / multi (多面板)")
    print("  New: /api/pattern_search/<ticker>")
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
