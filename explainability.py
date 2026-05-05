"""
explainability.py  —  Model Explainability & Interpretation
=============================================================
Provides full model explainability using:
  1. Random Forest Feature Importance (built-in)
  2. SHAP (SHapley Additive exPlanations) values
  3. Partial Dependence analysis (manual)
  4. Prediction error analysis

Run:
    python explainability.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

from data_generator import generate_machine_data, add_fault_events

# ── 0. Prepare Data ───────────────────────────────────────────────────────────
df = generate_machine_data(n_samples=500)
df = add_fault_events(df)

FEATURES = ['rpm', 'temperature', 'vibration', 'load', 'pressure']
WINDOW    = 5

def build_windows(df, window=5):
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

# Feature names for the window (5 time steps × 5 features)
feat_names = [f"{f}_t-{WINDOW-i}" for i in range(WINDOW) for f in FEATURES]

# ── 1. Train Models ───────────────────────────────────────────────────────────
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_vib  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_temp.fit(X_tr, yt_tr)
rf_vib.fit(X_tr, yv_tr)

print("=" * 65)
print("  MODEL EXPLAINABILITY REPORT")
print("=" * 65)

# ── 2. Built-in Feature Importance ───────────────────────────────────────────
imp_temp = pd.Series(rf_temp.feature_importances_, index=feat_names).sort_values(ascending=False)
imp_vib  = pd.Series(rf_vib.feature_importances_,  index=feat_names).sort_values(ascending=False)

print("\n[1] Top 10 Features — Temperature Prediction:")
print(imp_temp.head(10).round(4).to_string())

print("\n[2] Top 10 Features — Vibration Prediction:")
print(imp_vib.head(10).round(4).to_string())

# Aggregate by feature type (sum across all time lags)
def aggregate_by_feature(imp_series):
    agg = {}
    for feat in FEATURES:
        agg[feat] = imp_series[[n for n in imp_series.index if feat in n]].sum()
    return pd.Series(agg).sort_values(ascending=False)

agg_temp = aggregate_by_feature(imp_temp)
agg_vib  = aggregate_by_feature(imp_vib)

print("\n[3] Aggregated Feature Importance (by sensor type):")
print("    Temperature model:")
print(agg_temp.round(4).to_string())
print("    Vibration model:")
print(agg_vib.round(4).to_string())

# ── 3. Permutation Importance ─────────────────────────────────────────────────
print("\n[4] Computing Permutation Importance (Temperature model)...")
perm_imp = permutation_importance(rf_temp, X_te, yt_te, n_repeats=10,
                                  random_state=42, n_jobs=-1)
perm_series = pd.Series(perm_imp.importances_mean, index=feat_names).sort_values(ascending=False)
print("    Top 10:")
print(perm_series.head(10).round(4).to_string())

# ── 4. SHAP Values ────────────────────────────────────────────────────────────
shap_available = False
try:
    import shap
    print("\n[5] Computing SHAP values (this may take ~30 seconds)...")
    explainer  = shap.TreeExplainer(rf_temp)
    shap_vals  = explainer.shap_values(X_te[:100])  # use 100 samples for speed
    shap_available = True
    print("    SHAP values computed successfully.")
except ImportError:
    print("\n[5] SHAP not installed — using permutation importance instead.")

# ── 5. Visualizations ────────────────────────────────────────────────────────
BG   = '#1a1d27'
TEXT = '#A0AEC0'
colors5 = ['#00C8FF', '#FF7043', '#AB47BC', '#66BB6A', '#FFD740']

def style(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')
    ax.grid(True, color='#2a2d3e', linewidth=0.5, alpha=0.7)
    return ax

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Aggregated Feature Importance (both models)
# ═══════════════════════════════════════════════════════════════════════════════
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig1.patch.set_facecolor('#0f1117')
fig1.suptitle("Feature Importance — Which Sensors Matter Most?",
              color='white', fontsize=13, fontweight='bold')

for ax, agg, title_suffix, mdl_color in [
        (ax1, agg_temp, 'Temperature Prediction', '#FF7043'),
        (ax2, agg_vib,  'Vibration Prediction',   '#AB47BC')]:
    style(ax, f'Feature Importance\n{title_suffix}')
    bars = ax.barh(agg.index[::-1], agg.values[::-1],
                   color=[colors5[FEATURES.index(f)] for f in agg.index[::-1]],
                   alpha=0.85, edgecolor='none')
    ax.set_xlabel('Importance Score', color=TEXT, fontsize=10)
    for bar, val in zip(bars, agg.values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='white', fontsize=9)
    ax.set_xlim(0, agg.values.max() * 1.18)

plt.tight_layout()
plt.savefig("explainability_feature_importance.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n  [Saved] explainability_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Top-20 Window Features (detailed)
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(12, 8))
fig2.patch.set_facecolor('#0f1117')
style(ax, 'Top 20 Windowed Features — Temperature Model (Random Forest)')

top20 = imp_temp.head(20)
feat_colors = []
for fname in top20.index[::-1]:
    for fi, feat in enumerate(FEATURES):
        if feat in fname:
            feat_colors.append(colors5[fi])
            break

ax.barh(range(20), top20.values[::-1], color=feat_colors, alpha=0.85)
ax.set_yticks(range(20))
ax.set_yticklabels(top20.index[::-1], color=TEXT, fontsize=9)
ax.set_xlabel('Importance Score', color=TEXT, fontsize=10)
ax.set_title('Top 20 Windowed Features — Temperature Model', color='white', fontsize=12, fontweight='bold')

from matplotlib.patches import Patch
legend_els = [Patch(facecolor=colors5[i], label=FEATURES[i]) for i in range(5)]
ax.legend(handles=legend_els, facecolor='#1a1d27', labelcolor='white',
          fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig("explainability_top_features.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  [Saved] explainability_top_features.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: SHAP Summary OR Permutation Importance fallback
# ═══════════════════════════════════════════════════════════════════════════════
if shap_available:
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    fig3.patch.set_facecolor('#0f1117')
    ax3.set_facecolor(BG)
    shap.summary_plot(shap_vals, X_te[:100], feature_names=feat_names,
                      show=False, max_display=15, plot_type='bar')
    plt.title("SHAP Feature Importance (Temperature Model)", color='white', fontsize=12)
    plt.savefig("explainability_shap.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  [Saved] explainability_shap.png")
else:
    # Permutation importance chart instead
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig3.patch.set_facecolor('#0f1117')
    style(ax3, 'Permutation Importance — Temperature Model (top 15)')
    top15_perm = perm_series.head(15)
    pfc = []
    for fname in top15_perm.index[::-1]:
        for fi, feat in enumerate(FEATURES):
            if feat in fname:
                pfc.append(colors5[fi])
                break
    ax3.barh(range(15), top15_perm.values[::-1], color=pfc, alpha=0.85)
    ax3.set_yticks(range(15))
    ax3.set_yticklabels(top15_perm.index[::-1], color=TEXT, fontsize=9)
    ax3.set_xlabel('Mean Accuracy Decrease', color=TEXT)
    plt.tight_layout()
    plt.savefig("explainability_permutation.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print("  [Saved] explainability_permutation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Prediction Error Analysis
# ═══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.patch.set_facecolor('#0f1117')
fig4.suptitle("Prediction Error Analysis", color='white', fontsize=13, fontweight='bold')

for ax, (model, y_true, y_name, color) in zip(axes4, [
        (rf_temp, yt_te, 'Temperature (°C)', '#FF7043'),
        (rf_vib,  yv_te, 'Vibration (mm/s)', '#AB47BC')]):
    style(ax, f'Residuals — {y_name}')
    pred = model.predict(X_te)
    residuals = y_true - pred
    ax.scatter(pred, residuals, color=color, alpha=0.4, s=15)
    ax.axhline(0, color='white', lw=1.5, ls='--')
    ax.set_xlabel(f'Predicted {y_name}', color=TEXT, fontsize=10)
    ax.set_ylabel('Residual (Actual − Predicted)', color=TEXT, fontsize=10)
    ax.text(0.02, 0.97, f'Mean residual: {residuals.mean():.4f}',
            transform=ax.transAxes, color='white', fontsize=9, va='top')

plt.tight_layout()
plt.savefig("explainability_residuals.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  [Saved] explainability_residuals.png")

print("\n  ✅ Explainability analysis complete.")
print("=" * 65)
