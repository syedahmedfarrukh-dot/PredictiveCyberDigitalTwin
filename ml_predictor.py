"""
ml_predictor.py
AI/ML Prediction Layer — uses Random Forest to forecast future machine behavior.

Takes historical sensor readings as input and predicts:
  - Future temperature
  - Future vibration level
  - Failure probability (0–100%)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class MLPredictor:
    """
    Machine Learning predictor for the Digital Twin.
    
    Uses a sliding window of recent sensor readings to predict
    the machine's state N steps into the future.
    
    Model: Random Forest Regressor (robust, no scaling needed, handles noise well)
    """
    
    def __init__(self, window_size=5, forecast_steps=10):
        """
        Args:
            window_size    (int): Number of past time steps used as features
            forecast_steps (int): How many steps into the future to predict
        """
        self.window_size    = window_size
        self.forecast_steps = forecast_steps
        self.is_trained     = False
        self.scaler         = StandardScaler()
        
        # Separate model for each target variable
        self.models = {
            'temperature': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'vibration':   RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        }
        
        self.feature_columns = ['rpm', 'temperature', 'vibration', 'load', 'pressure']
        self.metrics = {}
    
    def _create_features(self, df):
        """
        Build sliding window feature matrix from time-series data.
        
        For each time step t, features are the sensor values at
        [t-window, t-window+1, ..., t-1].  Target is value at t+1.
        
        Returns:
            X (np.array): Feature matrix
            y_temp (np.array): Temperature targets
            y_vib  (np.array): Vibration targets
        """
        X, y_temp, y_vib = [], [], []
        
        values = df[self.feature_columns].values
        
        for i in range(self.window_size, len(values) - 1):
            window = values[i - self.window_size:i].flatten()
            X.append(window)
            y_temp.append(values[i + 1, 1])   # column 1 = temperature
            y_vib.append(values[i + 1, 2])    # column 2 = vibration
        
        return np.array(X), np.array(y_temp), np.array(y_vib)
    
    def train(self, df):
        """
        Train the prediction models on historical machine data.
        
        Args:
            df (pd.DataFrame): Historical sensor readings
        
        Returns:
            dict: Training performance metrics
        """
        print("[ML Predictor] Building training features...")
        X, y_temp, y_vib = self._create_features(df)
        
        X_train, X_test, y_temp_train, y_temp_test, y_vib_train, y_vib_test = (
            train_test_split(X, y_temp, y_vib, test_size=0.2, random_state=42)
        )
        
        # X is not needed for scaling in RF, but we keep scaler for consistency
        self.scaler.fit(X_train)
        
        print("[ML Predictor] Training temperature model...")
        self.models['temperature'].fit(X_train, y_temp_train)
        temp_pred = self.models['temperature'].predict(X_test)
        
        print("[ML Predictor] Training vibration model...")
        self.models['vibration'].fit(X_train, y_vib_train)
        vib_pred = self.models['vibration'].predict(X_test)
        
        # Compute metrics
        self.metrics = {
            'temperature': {
                'mae': round(mean_absolute_error(y_temp_test, temp_pred), 3),
                'r2':  round(r2_score(y_temp_test, temp_pred), 4),
            },
            'vibration': {
                'mae': round(mean_absolute_error(y_vib_test, vib_pred), 4),
                'r2':  round(r2_score(y_vib_test, vib_pred), 4),
            }
        }
        
        self.is_trained = True
        self._store_last_window(df)
        
        print("[ML Predictor] Training complete!")
        print(f"  Temperature  →  MAE: {self.metrics['temperature']['mae']} °C   R²: {self.metrics['temperature']['r2']}")
        print(f"  Vibration    →  MAE: {self.metrics['vibration']['mae']} mm/s  R²: {self.metrics['vibration']['r2']}")
        
        return self.metrics
    
    def _store_last_window(self, df):
        """Store the last window of data for online forecasting."""
        self.last_window = df[self.feature_columns].values[-self.window_size:]
    
    def predict_next(self, current_window=None):
        """
        Predict the next time step's temperature and vibration.
        
        Args:
            current_window: Optional override (uses stored window by default)
        
        Returns:
            dict with predicted temperature, vibration, and failure probability
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        window = current_window if current_window is not None else self.last_window
        features = window.flatten().reshape(1, -1)
        
        pred_temp = self.models['temperature'].predict(features)[0]
        pred_vib  = self.models['vibration'].predict(features)[0]
        
        # Failure probability: heuristic based on predicted values
        failure_prob = self._compute_failure_probability(pred_temp, pred_vib)
        
        return {
            'predicted_temperature': round(pred_temp, 2),
            'predicted_vibration':   round(pred_vib, 3),
            'failure_probability':   round(failure_prob, 1),
        }
    
    def forecast_future(self, steps=None):
        """
        Iteratively forecast multiple steps into the future.
        Each prediction feeds back as input for the next prediction.
        
        Args:
            steps (int): Number of future steps (default: self.forecast_steps)
        
        Returns:
            pd.DataFrame: Forecast results for each future step
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        steps = steps or self.forecast_steps
        window = self.last_window.copy()
        forecasts = []
        
        for step in range(1, steps + 1):
            features = window.flatten().reshape(1, -1)
            
            pred_temp = self.models['temperature'].predict(features)[0]
            pred_vib  = self.models['vibration'].predict(features)[0]
            failure_p = self._compute_failure_probability(pred_temp, pred_vib)
            
            forecasts.append({
                'step':                  step,
                'predicted_temperature': round(pred_temp, 2),
                'predicted_vibration':   round(pred_vib, 3),
                'failure_probability':   round(failure_p, 1),
            })
            
            # Slide window: drop oldest row, append predicted as newest
            new_row = window[-1].copy()
            new_row[1] = pred_temp   # column 1 = temperature
            new_row[2] = pred_vib    # column 2 = vibration
            window = np.vstack([window[1:], new_row])
        
        return pd.DataFrame(forecasts)
    
    def _compute_failure_probability(self, temperature, vibration):
        """
        Heuristic failure probability (0–100%) based on predicted values.
        
        Combines temperature and vibration severity into a single risk score.
        """
        # Temperature contribution (0 = safe, 1 = critical)
        temp_score = max(0, (temperature - 70) / (105 - 70))
        temp_score = min(temp_score, 1.0)
        
        # Vibration contribution (0 = safe, 1 = critical)
        vib_score = max(0, (vibration - 2.0) / (6.5 - 2.0))
        vib_score = min(vib_score, 1.0)
        
        # Weighted combination
        failure_prob = (0.60 * temp_score + 0.40 * vib_score) * 100
        return min(failure_prob, 99.9)


if __name__ == "__main__":
    from data_generator import generate_machine_data, add_fault_events
    
    print("=== ML Predictor Test ===\n")
    
    df = generate_machine_data(n_samples=500)
    df = add_fault_events(df)
    
    predictor = MLPredictor(window_size=5, forecast_steps=15)
    predictor.train(df)
    
    print("\n--- 15-Step Forecast ---")
    forecast = predictor.forecast_future(steps=15)
    print(forecast.to_string(index=False))
