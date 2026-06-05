"""
Lightweight HTTP dashboard server for live training metrics.

Starts a Flask server that serves:
  - /         — HTML dashboard with auto-refreshing plots
  - /api/metrics  — JSON endpoint with current metrics
  - /api/analytics — PNG renders of analytics charts

Usage:
    python dashboard_server.py --port 8080
    python dashboard_server.py --csv logs/metrics.csv --port 5000
"""

import os
import sys
import json
import argparse
import csv
import io
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, jsonify, send_file, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>RL Car Training Dashboard</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="30">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 10px; font-size: 24px; }
        .stats { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
        .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                     padding: 16px 24px; min-width: 140px; }
        .stat-card .label { font-size: 12px; color: #8b949e; text-transform: uppercase; }
        .stat-card .value { font-size: 28px; font-weight: bold; color: #58a6ff; }
        .chart-container { background: #161b22; border: 1px solid #30363d;
                          border-radius: 8px; padding: 20px; margin: 20px 0; }
        img { max-width: 100%; border-radius: 4px; }
        .status { font-size: 13px; color: #8b949e; margin: 10px 0; }
        .refresh { color: #3fb950; }
    </style>
</head>
<body>
    <h1>RL Car Training Dashboard</h1>
    <div class="status">Auto-refreshes every 30s | {{ status }}</div>

    <div class="stats">
        <div class="stat-card">
            <div class="label">Total Episodes</div>
            <div class="value">{{ total_episodes }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Best Reward</div>
            <div class="value">{{ best_reward }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Avg Reward (last 100)</div>
            <div class="value">{{ avg_reward }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Latest Reward</div>
            <div class="value">{{ latest_reward }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Crash Rate</div>
            <div class="value">{{ crash_rate }}</div>
        </div>
        <div class="stat-card">
            <div class="label">Avg Laps</div>
            <div class="value">{{ avg_laps }}</div>
        </div>
    </div>

    <div class="chart-container">
        <img src="/api/chart/reward" alt="Reward chart" loading="lazy">
    </div>
    <div class="chart-container">
        <img src="/api/chart/dashboard" alt="Dashboard" loading="lazy">
    </div>

    <div class="status refresh">Last updated: {{ last_update }}</div>
</body>
</html>"""


def load_metrics_data(csv_path: str) -> dict:
    """Load the latest metrics from CSV."""
    if not os.path.exists(csv_path):
        return {
            "total_episodes": 0, "best_reward": 0, "avg_reward": 0,
            "latest_reward": 0, "crash_rate": "0%", "avg_laps": 0,
        }
    rewards = []
    crashes = []
    laps = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rewards.append(float(row.get("reward", 0)))
                crashes.append(float(row.get("crash_rate", 0)))
                laps.append(float(row.get("laps", 0)))
    except Exception:
        pass

    if not rewards:
        return {
            "total_episodes": 0, "best_reward": 0, "avg_reward": 0,
            "latest_reward": 0, "crash_rate": "0%", "avg_laps": 0,
        }

    total = len(rewards)
    return {
        "total_episodes": total,
        "best_reward": f"{max(rewards):.1f}",
        "avg_reward": f"{np.mean(rewards[-100:]):.1f}" if rewards else "0",
        "latest_reward": f"{rewards[-1]:.1f}",
        "crash_rate": f"{np.mean(crashes[-100:]) * 100:.1f}%" if crashes else "0%",
        "avg_laps": f"{np.mean(laps[-100:]):.1f}" if laps else "0",
    }


def _generate_dashboard_chart(csv_path: str) -> bytes:
    """Generate a multi-panel dashboard chart PNG in memory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(csv_path):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", fontsize=16)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    episodes, rewards, lengths, crashes, laps = [], [], [], [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            episodes.append(float(row.get("episode", len(episodes) + 1)))
            rewards.append(float(row.get("reward", 0)))
            lengths.append(float(row.get("length", 0)))
            crashes.append(float(row.get("crash_rate", 0)))
            laps.append(float(row.get("laps", 0)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#0d1117")
    colors = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff"]

    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#58a6ff")
        ax.grid(True, alpha=0.15, color="#c9d1d9")

    # Reward
    ax = axes[0, 0]
    ax.scatter(episodes, rewards, s=3, alpha=0.15, color=colors[0])
    if len(rewards) >= 20:
        w = min(20, len(rewards) // 5)
        if w > 0:
            cs = np.cumsum(np.insert(rewards, 0, 0))
            rm = (cs[w:] - cs[:-w]) / w
            ax.plot(episodes[w - 1:], rm, color=colors[1], linewidth=2)
    ax.set_title("Reward")

    # Length
    ax = axes[0, 1]
    ax.scatter(episodes, lengths, s=3, alpha=0.15, color=colors[1])
    ax.set_title("Episode Length")

    # Crash Rate
    ax = axes[1, 0]
    ax.plot(episodes, crashes, color=colors[2], linewidth=1, alpha=0.5)
    if len(crashes) >= 20:
        w = min(20, len(crashes) // 5)
        if w > 0:
            cs = np.cumsum(np.insert(crashes, 0, 0))
            cm = (cs[w:] - cs[:-w]) / w
            ax.plot(episodes[w - 1:], cm, color=colors[2], linewidth=2)
    ax.set_title("Crash Rate")
    ax.set_ylim(-0.02, 1.02)

    # Laps
    ax = axes[1, 1]
    ax.plot(episodes, laps, color=colors[3], linewidth=1, alpha=0.5)
    if len(laps) >= 20:
        w = min(20, len(laps) // 5)
        if w > 0:
            cs = np.cumsum(np.insert(laps, 0, 0))
            lm = (cs[w:] - cs[:-w]) / w
            ax.plot(episodes[w - 1:], lm, color=colors[3], linewidth=2)
    ax.set_title("Laps")

    fig.suptitle("Training Dashboard", color="#c9d1d9", fontsize=14, fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _generate_reward_chart(csv_path: str) -> bytes:
    """Generate a reward chart PNG in memory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(csv_path):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center", fontsize=16)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    episodes, rewards = [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            episodes.append(float(row.get("episode", len(episodes) + 1)))
            rewards.append(float(row.get("reward", 0)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor("#161b22")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#58a6ff")
    ax.grid(True, alpha=0.15, color="#c9d1d9")

    ax.scatter(episodes, rewards, s=3, alpha=0.15, color="#58a6ff", label="Episode reward")
    if len(rewards) >= 50:
        window = min(50, len(rewards) // 5)
        if window > 0:
            cumsum = np.cumsum(np.insert(rewards, 0, 0))
            rm = (cumsum[window:] - cumsum[:-window]) / window
            ax.plot(episodes[window - 1:], rm, color="#3fb950", linewidth=2,
                    label=f"Rolling mean (w={window})")
    ax.axhline(y=np.mean(rewards), color="#f0883e", linestyle="--", linewidth=1,
               label=f"Mean: {np.mean(rewards):.1f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Reward Progression")
    ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def create_app(csv_path: str, host: str = "0.0.0.0", port: int = 8080):
    """Create and configure the Flask application."""
    if not FLASK_AVAILABLE:
        print("Flask is not installed. Install with: pip install flask")
        return None

    app = Flask(__name__)

    @app.route("/")
    def index():
        metrics = load_metrics_data(csv_path)
        return render_template_string(
            HTML_TEMPLATE,
            **metrics,
            status="Connected" if metrics["total_episodes"] > 0 else "Waiting for data...",
            last_update=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    @app.route("/api/metrics")
    def api_metrics():
        import json
        return jsonify(load_metrics_data(csv_path))

    @app.route("/api/chart/<chart_name>")
    def api_chart(chart_name):
        if chart_name == "reward":
            png_data = _generate_reward_chart(csv_path)
            return send_file(io.BytesIO(png_data), mimetype="image/png")
        elif chart_name == "dashboard":
            png_data = _generate_dashboard_chart(csv_path)
            return send_file(io.BytesIO(png_data), mimetype="image/png")
        return "Chart not found", 404

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "csv_exists": os.path.exists(csv_path)})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training dashboard HTTP server")
    parser.add_argument("--csv", default="logs/metrics.csv", help="Path to metrics CSV")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    if not FLASK_AVAILABLE:
        print("Error: Flask is required. Install with: pip install flask")
        sys.exit(1)

    app = create_app(args.csv, args.host, args.port)
    if app:
        print(f"Dashboard server running at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
