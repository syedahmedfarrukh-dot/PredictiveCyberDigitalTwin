"""
baseline_model.py  —  Baseline vs Advanced Model Comparison
=============================================================
Implements and compares:
  BASELINE  : Linear Regression (simple, interpretable)
  ADVANCED  : Random Forest Regressor (our main ML model)

Targets: Temperature prediction & Vibration prediction
Metrics : MAE, RMSE, R²

Run:
    python baseline_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from data_generator import generate_machine_data, add_fault_events

# ── 0. Prepare Data ───────────────────────────────────────────────────────────
df = generate_machine_data(n_samples=500)
df = add_fault_events(df)

FEATURES = ['rpm', 'temperature', 'vibration', 'load', 'pressure']
WINDOW    = 5

def build_windows(df, window=5):
    """Sliding window feature matrix."""
    X, y_temp, y_vib = [], [], []
    vals = df[FEATURES].values
    for i in range(window, len(vals) - 1):
        X.append(vals[i - window:i].flatten())
        y_temp.append(vals[i + 1, 1])
        y_vib.append(vals[i + 1, 2])
    return np.array(X), np.array(y_temp), np.array(y_vib)

X, y_temp, y_vib = build_windows(df, WINDOW)
X_tr, X_te, yt_tr, yt_te, yv_tr, yv_te = train_test_split(
    X, y_temp, y_vib, test_size=0.2, random_state=42)

# Scale for Linear Regression
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# ── 1. Define Models ──────────────────────────────────────────────────────────
models = {
    'Linear Regression\n(Baseline)': {
        'temp': LinearRegression(),
        'vib':  LinearRegression(),
        'X_tr': X_tr_s, 'X_te': X_te_s,
        'color': '#00C8FF'
    },
    'Random Forest\n(Advanced)': {
        'temp': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'vib':  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'X_tr': X_tr, 'X_te': X_te,
        'color': '#66BB6A'
    },
}

# ── 2. Train & Evaluate ───────────────────────────────────────────────────────
results = {}

print("=" * 65)
print("  BASELINE vs ADVANCED MODEL COMPARISON")
print("=" * 65)

for name, cfg in models.items():
    label = name.replace('\n', ' ')
    r = {}
    for target_name, y_tr, y_te in [('temperature', yt_tr, yt_te),
                                     ('vibration',   yv_tr, yv_te)]:
        model = cfg['temp'] if target_name == 'temperature' else cfg['vib']
        model.fit(cfg['X_tr'], y_tr)
        pred = model.predict(cfg['X_te'])

        mae  = mean_absolute_error(y_te, pred)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        r2   = r2_score(y_te, pred)

        r[target_name] = {'mae': mae, 'rmse': rmse, 'r2': r2,
                           'pred': pred, 'true': y_te}

        print(f"\n  [{label}] — {target_name.capitalize()}")
        print(f"    MAE  : {mae:.4f}")
        print(f"    RMSE : {rmse:.4f}")
        print(f"    R²   : {r2:.4f}")

    results[name] = r

# ── 3. Cross-Validation on Temperature (Random Forest) ───────────────────────
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(rf_temp, X, y_temp, cv=5, scoring='r2')
print(f"\n  [Random Forest] 5-Fold CV R² (Temperature): "
      f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 4. Visualizations ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f1117')
fig.suptitle("Baseline vs Advanced Model — Comparison Report",
             color='white', fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.38)

BG   = '#1a1d27'
TEXT = '#A0AEC0'

def style(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_title(title, color='white', fontsize=10, fontweight='bold')
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')
    ax.grid(True, color='#2a2d3e', linewidth=0.5, alpha=0.7)
    return ax

model_labels = ['Linear Regression\n(Baseline)', 'Random Forest\n(Advanced)']
colors_bar   = ['#00C8FF', '#66BB6A']

# Row 0: Metric bar charts
metrics_map = {'MAE': 'mae', 'RMSE': 'rmse', 'R²': 'r2'}
for mi, (mname, mkey) in enumerate(metrics_map.items()):
    for ti, target in enumerate(['temperature', 'vibration']):
        ax = fig.add_subplot(gs[0, mi + (1 if ti == 1 and mi == 2 else 0) if mi < 2 else mi + ti])
        break

# Simpler: 4 bar charts in row 0
targets = ['temperature', 'vibration']
metric_pairs = [('MAE', 'mae'), ('R²', 'r2')]

for col_i, (target, (mname, mkey)) in enumerate(
        [(t, m) for t in targets for m in metric_pairs]):
    ax = fig.add_subplot(gs[0, col_i])
    style(ax, f'{target.capitalize()} — {mname}')
    vals = [results[ml][target][mkey] for ml in model_labels]
    bars = ax.bar(range(2), vals, color=colors_bar, alpha=0.85, width=0.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(['Baseline\n(LR)', 'Advanced\n(RF)'], color=TEXT, fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{val:.3f}', ha='center', color='white', fontsize=9, fontweight='bold')

# Row 1: Actual vs Predicted — Temperature
for mi, ml in enumerate(model_labels):
    ax = fig.add_subplot(gs[1, mi*2 : mi*2+2])
    style(ax, f'{ml.replace(chr(10)," ")} — Temperature: Actual vs Predicted')
    true = results[ml]['temperature']['true']
    pred = results[ml]['temperature']['pred']
    ax.plot(true[:80],  color='#FF7043', lw=1.5, label='Actual',    alpha=0.9)
    ax.plot(pred[:80],  color=colors_bar[mi], lw=1.5, label='Predicted', alpha=0.9, ls='--')
    ax.set_xlabel('Sample', color=TEXT, fontsize=9)
    ax.set_ylabel('Temperature (°C)', color=TEXT, fontsize=9)
    ax.legend(fontsize=8, facecolor=BG, labelcolor='white')

# Row 2: Actual vs Predicted — Vibration
for mi, ml in enumerate(model_labels):
    ax = fig.add_subplot(gs[2, mi*2 : mi*2+2])
    style(ax, f'{ml.replace(chr(10)," ")} — Vibration: Actual vs Predicted')
    true = results[ml]['vibration']['true']
    pred = results[ml]['vibration']['pred']
    ax.plot(true[:80],  color='#AB47BC', lw=1.5, label='Actual',    alpha=0.9)
    ax.plot(pred[:80],  color=colors_bar[mi], lw=1.5, label='Predicted', alpha=0.9, ls='--')
    ax.set_xlabel('Sample', color=TEXT, fontsize=9)
    ax.set_ylabel('Vibration (mm/s)', color=TEXT, fontsize=9)
    ax.legend(fontsize=8, facecolor=BG, labelcolor='white')

plt.savefig("baseline_comparison.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n  [Saved] baseline_comparison.png")
print("\n  ✅ Baseline comparison complete.")
print("=" * 65)
