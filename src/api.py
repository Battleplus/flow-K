from flask import Flask, jsonify, request, send_from_directory, send_file
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw, load_ticker, list_tickers
from src.patterns import add_indicators, detect_all, SIGNALS
from src.stats import compute_forward_returns, all_signal_stats
from src.chart import plot_kline_with_signals, plot_forward_distribution
from src.report import _judge_signal, _overall_judgment

app = Flask(__name__, static_folder=str(ROOT / 'web'), static_url_path='')

@app.route('/')
def index():
    return send_from_directory(str(ROOT / 'web'), 'index.html')

@app.route('/api/chart')
def api_chart():
    """Serve generated chart images by absolute path."""
    path = request.args.get('path', '')
    p = Path(path)
    if not p.exists():
        return ('Not found', 404)
    return send_file(str(p), mimetype='image/png')

@app.route('/api/tickers')
def api_tickers():
    raw = load_raw()
    tickers = list_tickers(raw)
    return jsonify(tickers)

@app.route('/api/analyze/<ticker>')
def api_analyze(ticker):
    lookback = request.args.get('lookback', 3.0, type=float)
    raw = load_raw()
    df = load_ticker(raw, ticker.upper())
    df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=int(lookback * 365))].copy()
    
    df = add_indicators(df)
    df = detect_all(df)
    df = compute_forward_returns(df)
    
    all_stats = all_signal_stats(df)
    latest = df.iloc[-1]
    
    current_signals = []
    for key in SIGNALS:
        col = f'signal_{key}'
        if col in df.columns and latest[col] == 1:
            current_signals.append(key)
    
    overall = _overall_judgment(all_stats)
    
    signal_details = []
    for key, (name, _) in SIGNALS.items():
        s = all_stats.get(key, {})
        judge = _judge_signal(s)
        signal_details.append({
            'key': key,
            'name': name,
            'total_occurrences': s.get('total_occurrences', 0),
            'judge': judge,
            'forward': s.get('forward', {}),
        })
    
    chart_path = plot_kline_with_signals(df, ticker.upper(), signal_cols=[f'signal_{k}' for k in SIGNALS])
    dist_charts = {}
    for key in SIGNALS:
        p = plot_forward_distribution(df, f'signal_{key}', ticker.upper(), forward_n=5)
        if p:
            dist_charts[key] = p
    
    return jsonify({
        'ticker': ticker.upper(),
        'data_start': str(df['Date'].min().date()),
        'data_end': str(df['Date'].max().date()),
        'records': len(df),
        'latest_close': float(latest['Adj Close']),
        'current_signals': current_signals,
        'overall': overall,
        'signals': signal_details,
        'chart_path': chart_path,
        'dist_charts': dist_charts,
    })

if __name__ == '__main__':
    import pandas as pd
    print('Starting K-line Strategy Dashboard...')
    print('Open http://localhost:5000 in your browser')
    app.run(host='0.0.0.0', port=5000, debug=True)
