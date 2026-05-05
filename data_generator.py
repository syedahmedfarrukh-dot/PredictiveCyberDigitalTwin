"""
data_generator.py
Generates synthetic sensor data for a mechanical machine (motor/pump system).
Simulates realistic behavior: RPM, Temperature, Vibration, Load, Pressure.
"""

import numpy as np
import pandas as pd


def generate_machine_data(n_samples=500, seed=42):
    """
    Generate synthetic time-series sensor data for a mechanical machine.
    
    Physics-based relationships:
      - Higher RPM  → Higher temperature + vibration
      - Higher load → Higher pressure + temperature
      - Random noise is added to simulate real sensor behavior
    
    Returns:
        pd.DataFrame with columns: time, rpm, temperature, vibration, load, pressure
    """
    np.random.seed(seed)
    
    time = np.arange(n_samples)
    
    # RPM: base 1500 with slow drift and random variation
    rpm = 1500 + 200 * np.sin(time / 50) + np.random.normal(0, 30, n_samples)
    rpm = np.clip(rpm, 800, 2200)
    
    # Load: random load cycles 20% to 90%
    load = 50 + 30 * np.sin(time / 80 + 1) + np.random.normal(0, 8, n_samples)
    load = np.clip(load, 20, 95)
    
    # Temperature: driven by RPM and Load (physics relationship)
    temperature = (
        40                              # base idle temperature
        + 0.015 * rpm                   # RPM contribution
        + 0.30 * load                   # load contribution
        + np.random.normal(0, 2, n_samples)  # sensor noise
    )
    temperature = np.clip(temperature, 35, 120)
    
    # Vibration: increases with RPM and randomly spikes (bearing wear simulation)
    vibration = (
        0.5
        + 0.002 * rpm
        + 0.01 * load
        + np.random.exponential(0.3, n_samples)  # occasional spikes
    )
    vibration = np.clip(vibration, 0.1, 8.0)
    
    # Pressure: driven by RPM and load
    pressure = (
        1.0
        + 0.001 * rpm
        + 0.02 * load
        + np.random.normal(0, 0.5, n_samples)
    )
    pressure = np.clip(pressure, 0.5, 6.0)
    
    df = pd.DataFrame({
        'time':        time,
        'rpm':         np.round(rpm, 1),
        'temperature': np.round(temperature, 2),
        'vibration':   np.round(vibration, 3),
        'load':        np.round(load, 1),
        'pressure':    np.round(pressure, 2)
    })
    
    return df


def add_fault_events(df):
    """
    Inject fault events into the data to train the ML model on failure patterns.
    Adds temperature spikes and vibration surges at certain time steps.
    """
    df = df.copy()
    
    # Overheating event: time steps 150-170
    df.loc[150:170, 'temperature'] += 25
    df.loc[150:170, 'vibration']   += 1.5
    
    # Bearing wear event: time steps 320-340
    df.loc[320:340, 'vibration'] += 3.0
    df.loc[320:340, 'temperature'] += 10
    
    # High load surge: time steps 420-440
    df.loc[420:440, 'load']        += 30
    df.loc[420:440, 'pressure']    += 1.5
    df.loc[420:440, 'temperature'] += 15
    
    return df


if __name__ == "__main__":
    df = generate_machine_data()
    df = add_fault_events(df)
    df.to_csv("machine_data.csv", index=False)
    print(f"[Data Generator] Generated {len(df)} samples.")
    print(df.head(10).to_string(index=False))
    print(f"\nTemperature range : {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")
    print(f"RPM range         : {df['rpm'].min():.0f} - {df['rpm'].max():.0f} RPM")
    print(f"Vibration range   : {df['vibration'].min():.2f} - {df['vibration'].max():.2f} mm/s")
