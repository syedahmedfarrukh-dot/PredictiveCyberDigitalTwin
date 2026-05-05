"""
digital_twin.py
The Digital Twin Model — a virtual representation of the physical machine.

This module simulates machine behavior using engineering equations.
It can be given any RPM and load, and it outputs the expected
temperature, vibration, and pressure — just like the real machine would.
"""


class DigitalTwin:
    """
    Virtual replica of a mechanical machine (motor/pump system).
    
    The twin uses simplified physics equations to simulate
    how the machine behaves under different operating conditions.
    """
    
    def __init__(self):
        # Machine physical constants
        self.base_temp       = 40.0   # idle temperature (°C)
        self.rpm_temp_coeff  = 0.015  # temperature rise per RPM unit
        self.load_temp_coeff = 0.30   # temperature rise per % load
        
        self.base_vib        = 0.5    # base vibration (mm/s)
        self.rpm_vib_coeff   = 0.002  # vibration increase per RPM unit
        
        self.base_pressure   = 1.0    # base pressure (bar)
        self.rpm_pres_coeff  = 0.001  # pressure increase per RPM
        self.load_pres_coeff = 0.02   # pressure increase per % load
        
        # Safe operating limits
        self.limits = {
            'temperature': {'warning': 90,  'critical': 105},
            'vibration':   {'warning': 4.0, 'critical': 6.5},
            'pressure':    {'warning': 4.5, 'critical': 5.5},
            'rpm':         {'warning': 1950, 'critical': 2100},
        }
        
        # State history (used by ML layer)
        self.history = []
    
    def simulate(self, rpm, load):
        """
        Simulate machine state for given RPM and load.
        
        Args:
            rpm  (float): Rotational speed in RPM
            load (float): Load percentage (0–100)
        
        Returns:
            dict: Simulated machine state with all sensor readings
        """
        # Physics-based calculation of each parameter
        temperature = (
            self.base_temp
            + self.rpm_temp_coeff * rpm
            + self.load_temp_coeff * load
        )
        
        vibration = (
            self.base_vib
            + self.rpm_vib_coeff * rpm
            + 0.01 * load
        )
        
        pressure = (
            self.base_pressure
            + self.rpm_pres_coeff * rpm
            + self.load_pres_coeff * load
        )
        
        state = {
            'rpm':         round(rpm, 1),
            'load':        round(load, 1),
            'temperature': round(temperature, 2),
            'vibration':   round(vibration, 3),
            'pressure':    round(pressure, 2),
        }
        
        self.history.append(state)
        return state
    
    def check_alerts(self, state):
        """
        Check simulated state against safe operating limits.
        Returns a list of alert messages.
        """
        alerts = []
        
        for param, value in state.items():
            if param not in self.limits:
                continue
            
            critical_limit = self.limits[param]['critical']
            warning_limit  = self.limits[param]['warning']
            
            if value >= critical_limit:
                alerts.append({
                    'level':   'CRITICAL',
                    'param':   param,
                    'value':   value,
                    'limit':   critical_limit,
                    'message': f"CRITICAL: {param.upper()} = {value} exceeds limit {critical_limit}"
                })
            elif value >= warning_limit:
                alerts.append({
                    'level':   'WARNING',
                    'param':   param,
                    'value':   value,
                    'limit':   warning_limit,
                    'message': f"WARNING : {param.upper()} = {value} approaching limit {warning_limit}"
                })
        
        return alerts
    
    def get_efficiency(self, state):
        """
        Estimate system efficiency (%) based on current operating conditions.
        Higher temperature and vibration reduce efficiency.
        """
        base_efficiency = 100.0
        
        # Efficiency penalty from overheating
        temp_excess = max(0, state['temperature'] - 70)
        temp_penalty = temp_excess * 0.5
        
        # Efficiency penalty from excess vibration
        vib_excess = max(0, state['vibration'] - 2.0)
        vib_penalty = vib_excess * 3.0
        
        efficiency = max(0, base_efficiency - temp_penalty - vib_penalty)
        return round(efficiency, 1)


if __name__ == "__main__":
    twin = DigitalTwin()
    
    print("=== Digital Twin Simulation ===\n")
    
    test_cases = [
        (1500, 50, "Normal operation"),
        (1800, 75, "High load"),
        (2000, 90, "Near critical"),
        (1200, 30, "Light load"),
    ]
    
    for rpm, load, label in test_cases:
        state = twin.simulate(rpm, load)
        alerts = twin.check_alerts(state)
        efficiency = twin.get_efficiency(state)
        
        print(f"[{label}]")
        print(f"  RPM={state['rpm']}, Load={state['load']}%")
        print(f"  Temperature : {state['temperature']} °C")
        print(f"  Vibration   : {state['vibration']} mm/s")
        print(f"  Pressure    : {state['pressure']} bar")
        print(f"  Efficiency  : {efficiency}%")
        if alerts:
            for a in alerts:
                print(f"  ⚠  {a['message']}")
        else:
            print(f"  ✓  All parameters normal")
        print()
