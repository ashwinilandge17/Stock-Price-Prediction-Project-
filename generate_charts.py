"""
generate_charts.py
==================
Place this file in your StockSense AI project folder (same level as predictor.py, utils.py, data_fetcher.py).
Then run:  python generate_charts.py

Generates 5 JPG images using REAL TCS.NS data fetched via yfinance
and your own predictor.py + utils.py code.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_fetcher import fetch_stock_data_live
from predictor import StockPredictor
from utils import calculate_technical_indicators

TICKER    = "TCS.NS"
PERIOD    = "1y"
INTERVAL  = "1d"
PRED_DAYS = 7
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

BG      = "#060a14"
PANEL   = "#0d1224"
BORDER  = "#1e3a5f"
ACCENT  = "#00d4ff"
GREEN   = "#10b981"
RED     = "#ef4444"
YELLOW  = "#f59e0b"
PURPLE  = "#8b5cf6"
ORANGE  = "#f97316"
PINK    = "#ec4899"
MUTED   = "#475569"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT, "xtick.color": SUBTEXT, "ytick.color": SUBTEXT,
    "text.color": TEXT, "grid.color": BORDER, "grid.linewidth": 0.5,
    "font.family": "DejaVu Sans", "font.size": 9,
})

def spine_clean(ax, tick_size=8):
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=tick_size)

def xtick_dates(ax, dates, n=7, size=8, rotation=30):
    idx = np.linspace(0, len(dates)-1, n, dtype=int)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(dates[i].strftime("%d %b '%y")) for i in idx],
                       fontsize=size, rotation=rotation, ha="right")

# ── fetch data ────────────────────────────────────────────────────────────────
print(f"[1/6] Fetching {TICKER} data ...")
raw_df = fetch_stock_data_live(TICKER, period=PERIOD, interval=INTERVAL)
if raw_df is None or raw_df.empty:
    print("ERROR: Could not fetch data. Check internet / yfinance.")
    sys.exit(1)

df    = calculate_technical_indicators(raw_df.copy())
dates = df.index
N     = len(df)
x     = np.arange(N)
print(f"  Got {N} rows  ({dates[0].date()} to {dates[-1].date()})")

# ════════════════════════════════════════════════════════════════════════════
# CHART 1 — Historical Stock Price Trend
# ════════════════════════════════════════════════════════════════════════════
print("[2/6] Chart 1: Historical Stock Price Trend ...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8),
                                gridspec_kw={"height_ratios": [3, 1]})
fig.patch.set_facecolor(BG)
fig.subplots_adjust(hspace=0.06, left=0.07, right=0.97, top=0.93, bottom=0.10)

for i in range(N):
    c = GREEN if df["Close"].iloc[i] >= df["Open"].iloc[i] else RED
    ax1.plot([x[i], x[i]], [df["Low"].iloc[i], df["High"].iloc[i]],
             color=c, linewidth=0.5, alpha=0.8)
    h = abs(df["Close"].iloc[i] - df["Open"].iloc[i]) or 0.3
    ax1.bar(x[i], h, bottom=min(df["Close"].iloc[i], df["Open"].iloc[i]),
            width=0.75, color=c, alpha=0.9, linewidth=0)

if "BB_Upper" in df.columns:
    ax1.fill_between(x, df["BB_Lower"], df["BB_Upper"], alpha=0.07, color=ACCENT)
    ax1.plot(x, df["BB_Upper"], color=ACCENT, lw=0.8, ls="--", alpha=0.55)
    ax1.plot(x, df["BB_Lower"], color=ACCENT, lw=0.8, ls="--", alpha=0.55)
    ax1.plot(x, df["BB_Mid"],   color=ACCENT, lw=0.6, ls=":",  alpha=0.4)

for col, clr, lw in [("MA20", YELLOW, 1.2), ("MA50", ORANGE, 1.2), ("MA200", PURPLE, 1.5)]:
    if col in df.columns:
        ax1.plot(x, df[col], color=clr, linewidth=lw, label=col, alpha=0.9)

last_close = float(df["Close"].iloc[-1])
ax1.axhline(last_close, color=TEXT, lw=0.6, ls=":", alpha=0.5)
ax1.text(N+0.5, last_close, f"Rs.{last_close:,.1f}", color=TEXT, fontsize=7.5, va="center")
ax1.set_ylabel("Price (Rs.)", color=TEXT, fontsize=10)
ax1.set_xlim(-1, N+1)
ax1.grid(True, axis="y", alpha=0.25); ax1.grid(False, axis="x")
ax1.legend(loc="upper left", fontsize=8, framealpha=0.25,
           facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
spine_clean(ax1); ax1.set_xticklabels([])

vol_colors = [GREEN if df["Close"].iloc[i] >= df["Open"].iloc[i] else RED for i in range(N)]
ax2.bar(x, df["Volume"]/1e6, color=vol_colors, alpha=0.7, width=0.8)
ax2.set_ylabel("Volume (M)", color=TEXT, fontsize=9)
ax2.set_xlim(-1, N+1)
ax2.grid(True, axis="y", alpha=0.2); ax2.grid(False, axis="x")
xtick_dates(ax2, dates); spine_clean(ax2)

p1 = os.path.join(OUT_DIR, "1_historical_stock_price_trend.jpg")
fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Saved -> {p1}")

# ════════════════════════════════════════════════════════════════════════════
# CHART 2 — Technical Indicators Dashboard
# ════════════════════════════════════════════════════════════════════════════
print("[3/6] Chart 2: Technical Indicators Dashboard ...")
fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.32,
                       left=0.08, right=0.97, top=0.95, bottom=0.08)

ax = fig.add_subplot(gs[0, :])
ax.set_facecolor(PANEL)
if "RSI" in df.columns:
    ax.fill_between(x, df["RSI"], 70, where=(df["RSI"]>70), alpha=0.22, color=RED)
    ax.fill_between(x, df["RSI"], 30, where=(df["RSI"]<30), alpha=0.22, color=GREEN)
    ax.plot(x, df["RSI"], color=YELLOW, lw=1.2)
    ax.axhline(70, color=RED,   lw=0.8, ls="--", alpha=0.65, label="Overbought 70")
    ax.axhline(30, color=GREEN, lw=0.8, ls="--", alpha=0.65, label="Oversold 30")
    ax.axhline(50, color=MUTED, lw=0.5, ls=":")
    ax.set_ylim(0, 100)
ax.set_ylabel("RSI (14)", color=TEXT, fontsize=9)
ax.set_xlim(-1, N+1)
ax.legend(fontsize=7.5, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, axis="y", alpha=0.2); spine_clean(ax); ax.set_xticklabels([])

axm = fig.add_subplot(gs[1, :])
axm.set_facecolor(PANEL)
if "MACD" in df.columns:
    hc = [GREEN if v >= 0 else RED for v in df["MACD_Hist"]]
    axm.bar(x, df["MACD_Hist"], color=hc, alpha=0.6, width=0.8)
    axm.plot(x, df["MACD"],        color=ACCENT, lw=1.2, label="MACD")
    axm.plot(x, df["MACD_Signal"], color=ORANGE, lw=1.0, ls="--", label="Signal")
    axm.axhline(0, color=MUTED, lw=0.5, ls=":")
axm.set_ylabel("MACD", color=TEXT, fontsize=9)
axm.set_xlim(-1, N+1)
axm.legend(fontsize=7.5, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
axm.grid(True, axis="y", alpha=0.2); spine_clean(axm); axm.set_xticklabels([])

axs = fig.add_subplot(gs[2, 0])
axs.set_facecolor(PANEL)
if "Stoch_K" in df.columns:
    axs.fill_between(x, df["Stoch_K"], 80, where=(df["Stoch_K"]>80), alpha=0.2, color=RED)
    axs.fill_between(x, df["Stoch_K"], 20, where=(df["Stoch_K"]<20), alpha=0.2, color=GREEN)
    axs.plot(x, df["Stoch_K"], color=PURPLE, lw=1.1, label="%K")
    axs.plot(x, df["Stoch_D"], color=PINK,   lw=1.0, ls="--", label="%D")
    axs.axhline(80, color=RED,   lw=0.7, ls="--", alpha=0.6)
    axs.axhline(20, color=GREEN, lw=0.7, ls="--", alpha=0.6)
    axs.set_ylim(0, 100)
axs.set_ylabel("Stochastic", color=TEXT, fontsize=9)
axs.set_xlim(-1, N+1)
axs.legend(fontsize=7.5, framealpha=0.2, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
axs.grid(True, axis="y", alpha=0.2); spine_clean(axs)
xtick_dates(axs, dates, n=4)

axb = fig.add_subplot(gs[2, 1])
axb.set_facecolor(PANEL)
if "BB_Width" in df.columns:
    axb.fill_between(x, df["BB_Width"], alpha=0.28, color=ACCENT)
    axb.plot(x, df["BB_Width"], color=ACCENT, lw=1.2)
axb.set_ylabel("BB Width", color=TEXT, fontsize=9)
axb.set_xlim(-1, N+1)
axb.grid(True, axis="y", alpha=0.2); spine_clean(axb)
xtick_dates(axb, dates, n=4)

p2 = os.path.join(OUT_DIR, "2_technical_indicators_dashboard.jpg")
fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Saved -> {p2}")

# ════════════════════════════════════════════════════════════════════════════
# Train all models
# ════════════════════════════════════════════════════════════════════════════
print("[4/6] Training models (takes ~30s) ...")
model_keys = ["Linear Regression", "Random Forest", "Gradient Boosting", "All Models Ensemble"]
results    = {}
for mk in model_keys:
    p = StockPredictor(model_type=mk)
    preds, conf, met = p.predict(raw_df.copy(), pred_days=PRED_DAYS)
    results[mk] = (preds, conf, met)
    print(f"  {mk}: conf={conf}%  MAPE={met.get('mape',0):.2f}%")

best_preds, best_conf, best_metrics = results["All Models Ensemble"]
if best_preds is None:
    best_preds, best_conf, best_metrics = results["Random Forest"]

last_price    = float(raw_df["Close"].iloc[-1])
future_dates_pd = pd.bdate_range(raw_df.index[-1] + pd.Timedelta(days=1), periods=PRED_DAYS)

# ════════════════════════════════════════════════════════════════════════════
# CHART 3 — Actual vs Predicted
# ════════════════════════════════════════════════════════════════════════════
print("[5/6] Chart 3: Actual vs Predicted ...")
HIST_SHOW   = 60
actual_vals = raw_df["Close"].values[-HIST_SHOW:].astype(float)
x_hist      = np.arange(HIST_SHOW)

fig, ax = plt.subplots(figsize=(15, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
ax.plot(x_hist, actual_vals, color=TEXT, lw=1.6, label="Actual Price", alpha=0.95)
ax.axvline(HIST_SHOW-1, color=YELLOW, lw=1.0, ls=":", alpha=0.7)

if best_preds:
    x_fut  = np.arange(HIST_SHOW-1, HIST_SHOW-1+PRED_DAYS)
    bridge = [actual_vals[-1]] + list(best_preds)
    x_br   = np.arange(HIST_SHOW-1, HIST_SHOW-1+len(bridge))
    ci     = [abs(p - last_price)*0.04 + 2 + i*0.5 for i, p in enumerate(best_preds)]
    ax.fill_between(x_fut,
                    [best_preds[i]-ci[i] for i in range(PRED_DAYS)],
                    [best_preds[i]+ci[i] for i in range(PRED_DAYS)],
                    alpha=0.15, color=ACCENT)
    ax.plot(x_br, bridge, color=ACCENT, lw=1.8, ls="--",
            label=f"Ensemble Forecast (conf {best_conf}%)",
            marker="o", markersize=5, markerfacecolor=ACCENT,
            markeredgecolor=BG, markeredgewidth=1.2, alpha=0.92)

ax.text(HIST_SHOW+0.2,
        min(actual_vals)*1.0005,
        "Prediction Zone", color=YELLOW, fontsize=8, alpha=0.8)

tick_pos  = list(range(0, HIST_SHOW, 10)) + [HIST_SHOW-1]
all_lbl   = [raw_df.index[-HIST_SHOW+i].strftime("%d %b") for i in range(HIST_SHOW)]
ax.set_xticks(tick_pos)
ax.set_xticklabels([all_lbl[i] for i in tick_pos], fontsize=8, rotation=20, ha="right")
ax.set_ylabel("Price (Rs.)", color=TEXT, fontsize=10)
ax.legend(fontsize=9, framealpha=0.25, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, axis="y", alpha=0.22); ax.grid(False, axis="x")
ax.set_xlim(-1, HIST_SHOW+PRED_DAYS+1)
spine_clean(ax)
fig.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.12)

p3 = os.path.join(OUT_DIR, "3_actual_vs_predicted_stock_price.jpg")
fig.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Saved -> {p3}")

# ════════════════════════════════════════════════════════════════════════════
# CHART 4 — 7-Day Forecast
# ════════════════════════════════════════════════════════════════════════════
print("[6/6] Chart 4: 7-Day Forecast ...")
H30       = 30
hist30    = raw_df["Close"].values[-H30:].astype(float)

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
ax.plot(np.arange(H30), hist30, color=TEXT, lw=1.8, label="Historical (30d)", alpha=0.95)
ax.axvline(H30-1, color=MUTED, lw=1.0, ls=":", alpha=0.6)

if best_preds:
    x_fut  = np.arange(H30-1, H30-1+PRED_DAYS)
    bridge = [hist30[-1]] + list(best_preds)
    x_br   = np.arange(H30-1, H30-1+len(bridge))
    ci     = [abs(p-last_price)*0.04 + 3 + i*1.5 for i, p in enumerate(best_preds)]
    ax.fill_between(x_fut,
                    [best_preds[i]-ci[i] for i in range(PRED_DAYS)],
                    [best_preds[i]+ci[i] for i in range(PRED_DAYS)],
                    alpha=0.15, color=ACCENT, label="Confidence Band")
    ax.plot(x_br, bridge, color=ACCENT, lw=2.0, ls="--",
            label="Ensemble Forecast",
            marker="o", markersize=6, markerfacecolor=ACCENT,
            markeredgecolor=BG, markeredgewidth=1.5)
    for i, (fp, fd) in enumerate(zip(best_preds, future_dates_pd)):
        sig   = "BUY"  if fp > last_price*1.001 else ("SELL" if fp < last_price*0.999 else "HOLD")
        clr   = GREEN  if sig == "BUY"           else (RED   if sig == "SELL"           else YELLOW)
        xi    = H30 + i
        offset = 18 + (i % 2)*14
        ax.annotate(f"Rs.{fp:,.0f}\n{sig}",
                    xy=(xi, fp), xytext=(0, offset), textcoords="offset points",
                    ha="center", fontsize=7.5, color=clr, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=clr, alpha=0.45, lw=0.8))

hist_lbl = [raw_df.index[-H30+i].strftime("%b %d") for i in range(H30)]
fut_lbl  = [d.strftime("%b %d") for d in future_dates_pd]
all_lbl  = hist_lbl + fut_lbl
tick_pos = list(range(0, H30, 5)) + list(range(H30, H30+PRED_DAYS))
ax.set_xticks(tick_pos)
ax.set_xticklabels([all_lbl[i] for i in tick_pos], fontsize=7.5, rotation=30, ha="right")
ax.set_xlim(-1, H30+PRED_DAYS+1)
ax.set_ylabel("Price (Rs.)", color=TEXT, fontsize=10)
ax.legend(fontsize=9, framealpha=0.25, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
ax.grid(True, axis="y", alpha=0.22); ax.grid(False, axis="x")
spine_clean(ax)
fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.13)

p4 = os.path.join(OUT_DIR, "4_future_7day_forecast.jpg")
fig.savefig(p4, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Saved -> {p4}")

# ════════════════════════════════════════════════════════════════════════════
# CHART 5 — Model Performance Comparison
# ════════════════════════════════════════════════════════════════════════════
print("[7/6] Chart 5: Model Performance Comparison ...")
bar_colors  = [ACCENT, GREEN, ORANGE, PURPLE]
xlabels     = ["Linear\nRegression", "Random\nForest", "Gradient\nBoosting", "Ensemble"]
rmse_vals   = [results[k][2].get("rmse", 0) for k in model_keys]
mape_vals   = [results[k][2].get("mape", 0) for k in model_keys]
r2_vals     = [results[k][2].get("r2",   0) for k in model_keys]
conf_vals   = [float(results[k][1])          for k in model_keys]
xp          = np.arange(4); bw = 0.55

def bpanel(ax_, vals, ylabel, fmt, best_min=True):
    ax_.set_facecolor(PANEL)
    bars = ax_.bar(xp, vals, width=bw, color=bar_colors, alpha=0.85, zorder=3)
    mx = max(abs(v) for v in vals) or 1
    for bar_, v in zip(bars, vals):
        ax_.text(bar_.get_x()+bar_.get_width()/2,
                 bar_.get_height()+mx*0.015,
                 fmt.format(v), ha="center", va="bottom",
                 fontsize=9, color=TEXT, fontweight="bold")
    ax_.set_xticks(xp); ax_.set_xticklabels(xlabels, fontsize=8)
    ax_.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax_.set_ylim(0, mx*1.28)
    ax_.grid(True, axis="y", alpha=0.2); spine_clean(ax_)
    bi = vals.index(min(vals)) if best_min else vals.index(max(vals))
    bars[bi].set_edgecolor(YELLOW); bars[bi].set_linewidth(2.2)

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor(BG)
gs5 = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.35,
                        left=0.08, right=0.97, top=0.93, bottom=0.10)
bpanel(fig.add_subplot(gs5[0,0]), rmse_vals, "RMSE (Rs.)",   "{:.4f}",  best_min=True)
bpanel(fig.add_subplot(gs5[0,1]), mape_vals, "MAPE (%)",     "{:.2f}%", best_min=True)
bpanel(fig.add_subplot(gs5[1,0]), r2_vals,   "R2 Score",     "{:.4f}",  best_min=False)
bpanel(fig.add_subplot(gs5[1,1]), conf_vals, "Confidence (%)", "{:.0f}%", best_min=False)
fig.text(0.5, 0.01, "Gold border = best model per metric",
         ha="center", color=YELLOW, fontsize=8, alpha=0.8)

p5 = os.path.join(OUT_DIR, "5_model_performance_comparison.jpg")
fig.savefig(p5, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Saved -> {p5}")

print("\nAll 5 charts saved to:", OUT_DIR)
