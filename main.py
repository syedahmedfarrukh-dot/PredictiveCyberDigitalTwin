"""
main.py  —  Predictive Cyber Digital Twin
==========================================
HOW THIS WORKS:
  1. System auto-generates training data (500 samples)
  2. ML model trains on that data
  3. You enter your machine sensor values
  4. System predicts future behavior and gives health report

Usage:
    python main.py

Requirements:
    pip install numpy pandas scikit-learn matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from data_generator import generate_machine_data, add_fault_events
from digital_twin    import DigitalTwin
from ml_predictor    import MLPredictor


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def get_float(prompt, min_val, max_val):
    while True:
        try:
            val = float(input(f"    {prompt}: ").strip())
            if min_val <= val <= max_val:
                return val
            print(f"    Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("    Please enter a number (e.g. 1500 or 75.5)")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Generate Training Data and Train ML Model
# ══════════════════════════════════════════════════════════════════════════════

section("PREDICTIVE CYBER DIGITAL TWIN — Starting Up")

print("\n  [1/2] Generating training data...")
df = generate_machine_data(n_samples=500)
df = add_fault_events(df)
df.to_csv("machine_data.csv", index=False)
print("        500 machine samples generated.")

print("\n  [2/2] Training AI prediction model...")
predictor = MLPredictor(window_size=5, forecast_steps=20)
metrics   = predictor.train(df)
print(f"        Temperature model accuracy : R2 = {metrics['temperature']['r2']}")
print(f"        Vibration model accuracy   : R2 = {metrics['vibration']['r2']}")
print("\n  System ready. Please enter your machine readings below.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — User enters sensor values
# ══════════════════════════════════════════════════════════════════════════════

section("Enter Your Machine Sensor Values")

print("""
  Enter the CURRENT readings from your machine sensors.

  Acceptable ranges:
    RPM         :  800  - 2200   rpm
    Temperature :   35  - 120    C
    Vibration   :  0.1  - 8.0    mm/s
    Load        :   20  - 95     %
    Pressure    :  0.5  - 6.0    bar
""")

user_rpm  = get_float("RPM          (800  - 2200)",     800,  2200)
user_temp = get_float("Temperature  (35   - 120  C)",    35,   120)
user_vib  = get_float("Vibration    (0.1  - 8.0 mm/s)",  0.1,  8.0)
user_load = get_float("Load         (20   - 95   %)",    20,    95)
user_pres = get_float("Pressure     (0.5  - 6.0 bar)",   0.5,  6.0)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Digital Twin evaluation
# ══════════════════════════════════════════════════════════════════════════════

twin       = DigitalTwin()
twin_state = twin.simulate(user_rpm, user_load)

actual_state = {
    'rpm':         user_rpm,
    'load':        user_load,
    'temperature': user_temp,
    'vibration':   user_vib,
    'pressure':    user_pres,
}

alerts     = twin.check_alerts(actual_state)
efficiency = twin.get_efficiency(actual_state)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — ML Forecast
# ══════════════════════════════════════════════════════════════════════════════

user_row    = np.array([user_rpm, user_temp, user_vib, user_load, user_pres])
user_window = np.tile(user_row, (predictor.window_size, 1))
predictor.last_window = user_window

forecast = predictor.forecast_future(steps=20)

max_fp   = forecast['failure_probability'].max()
avg_fp   = forecast['failure_probability'].mean()
max_temp = forecast['predicted_temperature'].max()
max_vib  = forecast['predicted_vibration'].max()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Determine danger level
# ══════════════════════════════════════════════════════════════════════════════

if max_fp >= 60 or user_temp >= 105 or user_vib >= 6.5 or user_pres >= 5.5:
    danger_level = "CRITICAL"
    danger_icon  = "!!!"
    action       = "STOP MACHINE — Immediate inspection required."
    safe_pct     = 0
    level_note   = "Machine is in a dangerous state. Risk of breakdown is high."
elif max_fp >= 30 or user_temp >= 90 or user_vib >= 4.0 or user_pres >= 4.5:
    danger_level = "WARNING"
    danger_icon  = "***"
    action       = "Schedule maintenance within 24-48 hours."
    safe_pct     = max(0, round(100 - avg_fp))
    level_note   = "Machine is under stress. Some parameters need attention."
else:
    danger_level = "NORMAL"
    danger_icon  = ">>>"
    action       = "Machine is healthy. Continue regular monitoring."
    safe_pct     = max(0, round(100 - avg_fp))
    level_note   = "All parameters are within acceptable range."

# Build issues list with clear cause and explanation
issues = []
if user_temp >= 105:
    issues.append(("CRITICAL", "OVERHEATING",
        f"Temperature {user_temp}C exceeded critical limit (105C). Risk of thermal damage."))
elif user_temp >= 90:
    issues.append(("WARNING",  "HIGH TEMPERATURE",
        f"Temperature {user_temp}C above warning limit (90C). Check cooling system."))

if user_vib >= 6.5:
    issues.append(("CRITICAL", "SEVERE VIBRATION",
        f"Vibration {user_vib} mm/s exceeded critical limit (6.5). Check bearings immediately."))
elif user_vib >= 4.0:
    issues.append(("WARNING",  "HIGH VIBRATION",
        f"Vibration {user_vib} mm/s above warning limit (4.0). Possible bearing wear or imbalance."))

if user_pres >= 5.5:
    issues.append(("CRITICAL", "OVER PRESSURE",
        f"Pressure {user_pres} bar exceeded critical limit (5.5). Risk of seal or pipe failure."))
elif user_pres >= 4.5:
    issues.append(("WARNING",  "HIGH PRESSURE",
        f"Pressure {user_pres} bar above warning limit (4.5). Monitor pressure relief valve."))

if user_rpm >= 2100:
    issues.append(("CRITICAL", "OVER SPEED",
        f"RPM {user_rpm} exceeded critical limit (2100). Risk of mechanical failure."))
elif user_rpm >= 1950:
    issues.append(("WARNING",  "HIGH RPM",
        f"RPM {user_rpm} above warning limit (1950). Consider reducing operating speed."))

if max_temp >= 90:
    issues.append(("FORECAST", "TEMPERATURE RISING",
        f"AI predicts temperature will reach {max_temp:.1f}C within next 20 steps. Trend is worsening."))

if max_vib >= 4.0:
    issues.append(("FORECAST", "VIBRATION RISING",
        f"AI predicts vibration will reach {max_vib:.2f} mm/s within next 20 steps. Degradation detected."))

if max_fp >= 60:
    issues.append(("FORECAST", "HIGH FAILURE RISK",
        f"AI predicts {max_fp:.1f}% peak failure probability. Breakdown likely if not addressed."))
elif max_fp >= 30:
    issues.append(("FORECAST", "MODERATE FAILURE RISK",
        f"AI predicts {avg_fp:.1f}% average failure probability across next 20 steps."))


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL OUTPUT — Final Health Report
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n")
print("  " + "=" * 58)
print(f"  {danger_icon}        MACHINE HEALTH REPORT        {danger_icon}")
print("  " + "=" * 58)

# Block 1: Current sensor readings with status
print(f"""
  CURRENT SENSOR READINGS
  {"─" * 54}
  {"Parameter":<16} {"Value":<16} {"Status"}
  {"─" * 54}""")

sensor_rows = [
    ("RPM",         f"{user_rpm:.0f} rpm",    user_rpm,           1950, 2100),
    ("Temperature", f"{user_temp:.1f} C",     user_temp,          90,   105 ),
    ("Vibration",   f"{user_vib:.2f} mm/s",   user_vib,           4.0,  6.5 ),
    ("Load",        f"{user_load:.0f} %",     user_load,          80,   95  ),
    ("Pressure",    f"{user_pres:.1f} bar",   user_pres,          4.5,  5.5 ),
    ("Efficiency",  f"{efficiency:.1f} %",    100 - efficiency,   20,   40  ),
]

for name, display, val, warn_lim, crit_lim in sensor_rows:
    if val >= crit_lim:
        status = "[CRITICAL]"
    elif val >= warn_lim:
        status = "[WARNING] "
    else:
        status = "[  OK   ] "
    print(f"  {name:<16} {display:<16} {status}")

# Block 2: AI forecast numbers
print(f"""
  AI FORECAST SUMMARY  (next 20 steps ahead)
  {"─" * 54}
  Predicted max temperature  :  {max_temp:.1f} C
  Predicted max vibration    :  {max_vib:.3f} mm/s
  Average failure risk       :  {avg_fp:.1f} %
  Peak failure risk          :  {max_fp:.1f} %
  {"─" * 54}""")

# Block 3: Overall verdict
print(f"""  OVERALL VERDICT
  {"─" * 54}
  Danger Level    :  {danger_level}
  Machine Score   :  {safe_pct} / 100   (100 = perfectly safe)
  Assessment      :  {level_note}
  {"─" * 54}""")

# Block 4: Problems detected
if issues:
    print(f"  PROBLEMS DETECTED  ({len(issues)} issue{'s' if len(issues) > 1 else ''} found)\n")
    for i, (level, title, detail) in enumerate(issues, 1):
        marker = "[!!!] CRITICAL" if level == "CRITICAL" else "[***] WARNING " if level == "WARNING" else "[AI ] FORECAST"
        print(f"  {i}. {marker} — {title}")
        print(f"       {detail}")
        print()
else:
    print(f"  NO PROBLEMS DETECTED\n")
    print(f"  All sensor readings are within safe operating limits.\n")

# Block 5: Action
print("  " + "─" * 54)
print(f"  ACTION REQUIRED:")
print(f"  ==> {action}")
print("  " + "=" * 58)
print()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Optional multiple readings comparison
# ══════════════════════════════════════════════════════════════════════════════

section("Optional: Compare Multiple Readings")
print("  Do you want to enter more readings to compare trends?")

extra_readings = [dict(actual_state, label="Reading 1 (current)")]
add_more       = input("  Add more readings? (y/n): ").strip().lower()
reading_num    = 2

while add_more == 'y':
    print(f"\n  Enter Reading {reading_num}:\n")
    r_rpm  = get_float("RPM          (800  - 2200)",     800,  2200)
    r_temp = get_float("Temperature  (35   - 120  C)",    35,   120)
    r_vib  = get_float("Vibration    (0.1  - 8.0 mm/s)",  0.1,  8.0)
    r_load = get_float("Load         (20   - 95   %)",    20,    95)
    r_pres = get_float("Pressure     (0.5  - 6.0 bar)",   0.5,  6.0)
    extra_readings.append({
        'rpm': r_rpm, 'temperature': r_temp, 'vibration': r_vib,
        'load': r_load, 'pressure': r_pres,
        'label': f"Reading {reading_num}"
    })
    reading_num += 1
    add_more = input("\n  Add another? (y/n): ").strip().lower()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

section("Generating Dashboard")

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

ORANGE = '#FF7043'; PURPLE = '#AB47BC'; GREEN = '#66BB6A'
YELLOW = '#FFD740'; RED    = '#EF5350'; TEXT  = '#e0e0e0'
BG     = '#1a1d27'; CYAN   = '#00E5FF'

def sa(ax, title):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.title.set_color(TEXT); ax.title.set_fontsize(10); ax.title.set_fontweight('bold')
    ax.set_title(title)
    for sp in ax.spines.values(): sp.set_edgecolor('#333355')
    ax.grid(True, color='#2a2d3e', linewidth=0.5, alpha=0.7)
    ax.yaxis.label.set_color(TEXT); ax.xaxis.label.set_color(TEXT)
    return ax

steps = forecast['step'].values

ax1 = fig.add_subplot(gs[0, :2])
sa(ax1, "Training Data Temperature (model learned from this)")
ax1.plot(df['time'], df['temperature'], color=ORANGE, lw=1, alpha=0.7, label='Historical Temp')
ax1.axhline(90,        color=YELLOW, lw=1.2, ls='--', alpha=0.8, label='Warning (90C)')
ax1.axhline(105,       color=RED,    lw=1.2, ls='--', alpha=0.8, label='Critical (105C)')
ax1.axhline(user_temp, color=CYAN,   lw=1.8, ls='-',             label=f'Your input ({user_temp}C)')
ax1.fill_between(df['time'], df['temperature'], 40, alpha=0.12, color=ORANGE)
ax1.set_xlabel("Time Step"); ax1.set_ylabel("C")
ax1.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333355', labelcolor=TEXT)

ax2 = fig.add_subplot(gs[0, 2])
sa(ax2, "Current Machine Status (% of safe limit)")
params     = ['Temp (C)', 'Vib (mm/s)', 'Load (%)', 'Pressure (bar)']
values     = [user_temp, user_vib, user_load, user_pres]
limits     = [105, 6.5, 95, 5.5]
pct        = [min(v/l*100, 100) for v, l in zip(values, limits)]
bclrs      = [RED if p > 90 else YELLOW if p > 75 else GREEN for p in pct]
bars       = ax2.barh(params, pct, color=bclrs, alpha=0.85)
ax2.axvline(75, color=YELLOW, lw=1, ls='--', alpha=0.7)
ax2.axvline(90, color=RED,    lw=1, ls='--', alpha=0.7)
ax2.set_xlim(0, 115)
for bar, val, u in zip(bars, values, ['C','mm/s','%','bar']):
    ax2.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
             f'{val}{u}', va='center', color=TEXT, fontsize=8)

ax3 = fig.add_subplot(gs[1, :2])
sa(ax3, "AI Forecast: Predicted Temperature (Next 20 Steps)")
ax3.plot(steps, forecast['predicted_temperature'], color=ORANGE, lw=2,
         marker='o', markersize=4, label='Predicted Temp')
ax3.axhline(user_temp, color=CYAN,   lw=1.5, ls='--', label=f'Your current ({user_temp}C)')
ax3.axhline(90,        color=YELLOW, lw=1.2, ls='--', alpha=0.8, label='Warning (90C)')
ax3.axhline(105,       color=RED,    lw=1.2, ls='--', alpha=0.8, label='Critical (105C)')
ax3.fill_between(steps, forecast['predicted_temperature'], user_temp, alpha=0.15, color=ORANGE)
ax3.set_xlabel("Future Steps Ahead"); ax3.set_ylabel("C")
ax3.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333355', labelcolor=TEXT)

ax4 = fig.add_subplot(gs[1, 2])
sa(ax4, "AI Forecast: Vibration")
ax4.plot(steps, forecast['predicted_vibration'], color=PURPLE, lw=2,
         marker='s', markersize=4, label='Predicted Vib')
ax4.axhline(user_vib, color=CYAN,   lw=1.5, ls='--', label=f'Current ({user_vib})')
ax4.axhline(4.0,      color=YELLOW, lw=1,   ls='--', alpha=0.8, label='Warning')
ax4.axhline(6.5,      color=RED,    lw=1,   ls='--', alpha=0.8, label='Critical')
ax4.set_xlabel("Future Steps Ahead"); ax4.set_ylabel("mm/s")
ax4.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333355', labelcolor=TEXT)

ax5 = fig.add_subplot(gs[2, :2])
sa(ax5, "AI Forecast: Failure Probability (%)")
bclr2 = [RED if p >= 60 else YELLOW if p >= 30 else GREEN for p in forecast['failure_probability']]
ax5.bar(steps, forecast['failure_probability'], color=bclr2, alpha=0.85, width=0.7)
ax5.axhline(60, color=RED,    lw=1.2, ls='--', alpha=0.8, label='High risk (60%)')
ax5.axhline(30, color=YELLOW, lw=1.2, ls='--', alpha=0.8, label='Moderate (30%)')
ax5.set_ylim(0, 105)
ax5.set_xlabel("Future Steps Ahead"); ax5.set_ylabel("%")
ax5.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333355', labelcolor=TEXT)

ax6 = fig.add_subplot(gs[2, 2])
sa(ax6, "Readings Comparison" if len(extra_readings) > 1 else "Your Input Summary")
if len(extra_readings) > 1:
    labels = [r['label'] for r in extra_readings]
    temps  = [r['temperature'] for r in extra_readings]
    vibs   = [r['vibration']   for r in extra_readings]
    x = range(len(labels))
    ax6.bar([i-0.2 for i in x], temps,              width=0.35, color=ORANGE, alpha=0.85, label='Temp (C)')
    ax6.bar([i+0.2 for i in x], [v*10 for v in vibs], width=0.35, color=PURPLE, alpha=0.85, label='Vib x10')
    ax6.set_xticks(list(x))
    ax6.set_xticklabels(labels, rotation=15, fontsize=7, color=TEXT)
    ax6.axhline(90, color=YELLOW, lw=1, ls='--', alpha=0.6)
    ax6.legend(fontsize=8, facecolor='#1a1d27', edgecolor='#333355', labelcolor=TEXT)
else:
    info = [
        ('RPM',         f"{user_rpm:.0f}"),
        ('Temperature', f"{user_temp:.1f} C"),
        ('Vibration',   f"{user_vib:.2f} mm/s"),
        ('Load',        f"{user_load:.0f} %"),
        ('Pressure',    f"{user_pres:.1f} bar"),
        ('Efficiency',  f"{efficiency:.1f} %"),
    ]
    ax6.axis('off')
    y_pos = 0.88
    ax6.text(0.5, 0.97, "Your Machine Reading", transform=ax6.transAxes,
             color=TEXT, fontsize=10, fontweight='bold', ha='center', va='top')
    for label, val in info:
        ax6.text(0.05, y_pos, label, transform=ax6.transAxes, color='#aaaaaa', fontsize=9)
        ax6.text(0.95, y_pos, val,   transform=ax6.transAxes, color=GREEN,      fontsize=9, ha='right')
        y_pos -= 0.13

title_color = '#EF5350' if danger_level == 'CRITICAL' else '#FFD740' if danger_level == 'WARNING' else '#66BB6A'
fig.suptitle(
    f"Predictive Cyber Digital Twin  |  {danger_level}  |  RPM={user_rpm:.0f}, Temp={user_temp}C, Vib={user_vib}mm/s",
    color=title_color, fontsize=12, fontweight='bold', y=0.98
)

plt.savefig("dashboard.png", dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print("  Dashboard saved to: dashboard.png")