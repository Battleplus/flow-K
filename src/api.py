"""Flask API + 仪表盘服务端"""

from flask import Flask, jsonify, request, send_from_directory, send_file
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_data
from src.indicators import trend_analysis
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
        analysis = trend_analysis(df)
        chart_path = plot_kline_trend(df, ticker, analysis)
        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "data_start": str(df["Date"].min()),
            "data_end": str(df["Date"].max()),
            "records": len(df),
            "analysis": analysis,
            "chart_path": chart_path,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting K-line Trend Dashboard...")
    print("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=True)
