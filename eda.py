"""
eda.py  —  Exploratory Data Analysis
======================================
Performs comprehensive EDA on the machine sensor dataset:
  - Statistical summary
  - Distribution plots
  - Correlation heatmap
  - Time-series trends
  - Outlier detection (IQR)
  - Fault event visualization

Run:
    python eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from data_generator import generate_machine_data, add_fault_events

# ── 0. Generate / Load Data ──────────────────────────────────────────────────
df = generate_machine_data(n_samples=500)
df = add_fault_events(df)
df.to_csv("machine_data.csv", index=False)

features = ['rpm', 'temperature', 'vibration', 'load', 'pressure']

print("=" * 60)
print("  EXPLORATORY DATA ANALYSIS — Machine Sensor Dataset")
print("=" * 60)

# ── 1. Basic Info ─────────────────────────────────────────────────────────────
print(f"\n[1] Dataset Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"[2] Missing Values  : {df.isnull().sum().sum()} (none — synthetic dataset)")
print(f"[3] Duplicate Rows  : {df.duplicated().sum()}")
print("\n[4] Statistical Summary:")
print(df[features].describe().round(3).to_string())

# ── 2. IQR Outlier Detection ──────────────────────────────────────────────────
print("\n[5] Outlier Count (IQR method):")
outlier_counts = {}
for col in features:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    outlier_counts[col] = outliers
    print(f"    {col:<14}: {outliers} outliers")

# ── 3. Correlation Matrix ─────────────────────────────────────────────────────
print("\n[6] Pearson Correlation Matrix:")
print(df[features].corr().round(3).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Distribution + Boxplots
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 5, figsize=(18, 8))
fig1.patch.set_facecolor('#0f1117')
fig1.suptitle("EDA — Feature Distributions & Boxplots",
              color='white', fontsize=14, fontweight='bold', y=1.01)

colors = ['#00C8FF', '#FF7043', '#AB47BC', '#66BB6A', '#FFD740']

for i, (col, color) in enumerate(zip(features, colors)):
    # Histogram
    ax = axes[0, i]
    ax.set_facecolor('#1a1d27')
    ax.hist(df[col], bins=30, color=color, alpha=0.85, edgecolor='none')
    ax.axvline(df[col].mean(), color='white', lw=1.5, ls='--', label=f'μ={df[col].mean():.1f}')
    ax.set_title(col, color=color, fontsize=11, fontweight='bold')
    ax.tick_params(colors='#A0AEC0')
    ax.set_xlabel('Value', color='#A0AEC0', fontsize=9)
    ax.set_ylabel('Count', color='#A0AEC0', fontsize=9)
    ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='white')
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')

    # Boxplot
    ax2 = axes[1, i]
    ax2.set_facecolor('#1a1d27')
    bp = ax2.boxplot(df[col], patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.7),
                     medianprops=dict(color='white', linewidth=2),
                     whiskerprops=dict(color='#A0AEC0'),
                     capprops=dict(color='#A0AEC0'),
                     flierprops=dict(marker='o', color=color, markersize=4, alpha=0.5))
    ax2.set_title(f'{col} — Boxplot', color=color, fontsize=10)
    ax2.tick_params(colors='#A0AEC0')
    for sp in ax2.spines.values(): sp.set_edgecolor('#333355')

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("\n  [Saved] eda_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(8, 6))
fig2.patch.set_facecolor('#0f1117')
ax.set_facecolor('#1a1d27')

corr = df[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', ax=ax,
            cmap='coolwarm', center=0, linewidths=0.5,
            annot_kws={'size': 12, 'color': 'white'},
            cbar_kws={'shrink': 0.8})

ax.set_title("Feature Correlation Heatmap", color='white', fontsize=13, fontweight='bold', pad=15)
ax.tick_params(colors='white', labelsize=11)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  [Saved] eda_correlation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Time-Series Trends with Fault Windows
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
fig3.patch.set_facecolor('#0f1117')
fig3.suptitle("EDA — Time-Series Sensor Trends with Fault Events",
              color='white', fontsize=13, fontweight='bold')

fault_windows = [(150, 170, '#EF5350', 'Overheating'),
                 (320, 340, '#AB47BC', 'Bearing Wear'),
                 (420, 440, '#FFD740', 'Load Surge')]

for i, (col, color) in enumerate(zip(features, colors)):
    ax = axes3[i]
    ax.set_facecolor('#1a1d27')
    ax.plot(df['time'], df[col], color=color, lw=1.2, alpha=0.9)
    ax.set_ylabel(col, color=color, fontsize=10, fontweight='bold')
    ax.tick_params(colors='#A0AEC0', labelsize=8)
    ax.grid(True, color='#2a2d3e', linewidth=0.5, alpha=0.7)
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')

    # Shade fault windows
    for (start, end, fc, label) in fault_windows:
        ax.axvspan(start, end, alpha=0.25, color=fc)
        if i == 0:
            ax.text((start+end)/2, ax.get_ylim()[1]*0.98, label,
                    ha='center', va='top', color=fc, fontsize=8, fontweight='bold')

axes3[-1].set_xlabel("Time Step", color='#A0AEC0', fontsize=10)

# Legend for fault windows
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#EF5350', alpha=0.4, label='Overheating (t=150–170)'),
                   Patch(facecolor='#AB47BC', alpha=0.4, label='Bearing Wear (t=320–340)'),
                   Patch(facecolor='#FFD740', alpha=0.4, label='Load Surge (t=420–440)')]
fig3.legend(handles=legend_elements, loc='upper right', facecolor='#1a1d27',
            labelcolor='white', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("eda_timeseries.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  [Saved] eda_timeseries.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Pairplot-style scatter matrix (key pairs)
# ═══════════════════════════════════════════════════════════════════════════════
pairs = [('rpm', 'temperature'), ('rpm', 'vibration'),
         ('load', 'temperature'), ('load', 'pressure'),
         ('temperature', 'vibration'), ('vibration', 'pressure')]

fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9))
fig4.patch.set_facecolor('#0f1117')
fig4.suptitle("EDA — Key Feature Relationships (Scatter Matrix)",
              color='white', fontsize=13, fontweight='bold')

for ax, (xc, yc) in zip(axes4.flat, pairs):
    ax.set_facecolor('#1a1d27')
    scatter = ax.scatter(df[xc], df[yc], c=df['temperature'],
                         cmap='plasma', alpha=0.5, s=15)
    ax.set_xlabel(xc, color='#A0AEC0', fontsize=10)
    ax.set_ylabel(yc, color='#A0AEC0', fontsize=10)
    ax.tick_params(colors='#A0AEC0', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')
    r = df[[xc, yc]].corr().iloc[0, 1]
    ax.set_title(f'r = {r:.3f}', color='#FFD740', fontsize=10)

fig4.colorbar(scatter, ax=axes4.flat[-1], label='Temperature (°C)')
plt.tight_layout()
plt.savefig("eda_scatter.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  [Saved] eda_scatter.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Outlier Count Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(8, 4))
fig5.patch.set_facecolor('#0f1117')
ax5.set_facecolor('#1a1d27')
bars = ax5.bar(outlier_counts.keys(), outlier_counts.values(),
               color=['#00C8FF','#FF7043','#AB47BC','#66BB6A','#FFD740'], alpha=0.85)
ax5.set_title("Outlier Count per Feature (IQR Method)", color='white', fontsize=12, fontweight='bold')
ax5.set_ylabel("Number of Outliers", color='#A0AEC0')
ax5.tick_params(colors='#A0AEC0')
for sp in ax5.spines.values(): sp.set_edgecolor('#333355')
for bar, val in zip(bars, outlier_counts.values()):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha='center', color='white', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("eda_outliers.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  [Saved] eda_outliers.png")

print("\n  ✅ EDA complete — 5 figures saved.")
print("=" * 60)
