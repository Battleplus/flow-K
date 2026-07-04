"""Flask API + 仪表盘服务端 — 集成趋势线/斜率/曲线/形态信号"""

from flask import Flask, jsonify, request, send_from_directory, send_file
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_data
from src.indicators import trend_analysis
from src.patterns import add_indicators, detect_all, signal_summary, get_active_signals
from src.chart import plot_kline_trend

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
    try:
        df = fetch_data(ticker, period=period)

        # 基础趋势分析
        analysis = trend_analysis(df)

        # 信号系统分析
        df_sig = add_indicators(df)
        df_sig = detect_all(df_sig)
        sig = signal_summary(df_sig)
        active = get_active_signals(df_sig, recent_days=5)

        # 绘图
        chart_path = plot_kline_trend(df, ticker, analysis, sig)

        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "data_start": str(df["Date"].min()),
            "data_end": str(df["Date"].max()),
            "records": len(df),
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
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals/<ticker>")
def api_signals(ticker):
    """单独获取信号详情"""
    period = request.args.get("period", "1y")
    try:
        df = fetch_data(ticker, period=period)
        df = add_indicators(df)
        df = detect_all(df)
        sig = signal_summary(df)

        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "signals": sig,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting K-line Trend & Signal Dashboard...")
    print("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=True)
