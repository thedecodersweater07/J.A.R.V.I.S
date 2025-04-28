# intrusion_detection.py
# Intrusion Detection System (IDS) for the Jarvis security suite

import logging
import time
from collections import defaultdict

class IntrusionDetection:
    """Monitors network activity to detect suspicious behavior and potential intrusions"""
    
    def __init__(self):
        self.logger = logging.getLogger("IDS")
        self.running = False
        self.detection_rules = self._load_detection_rules()
        self.alerts = []
        self.baseline = {}
        self.anomaly_thresholds = {
            'traffic_volume': 1.5,  # 50% above baseline
            'connection_rate': 2.0,  # Double the baseline
            'error_rate': 3.0       # Triple the baseline
        }
        self.logger.info("Intrusion Detection System initialized")
    
    def start(self):
        """Start the IDS"""
        self.running = True
        self.logger.info("Intrusion Detection System started")
        return True
    
    def stop(self):
        """Stop the IDS"""
        self.running = False
        self.logger.info("Intrusion Detection System stopped")
        return True
    
    def _load_detection_rules(self):
        """Load detection rules - in a real system, would load from configuration files"""
        return {
            "repeated_auth_failures": {
                "threshold": 5,
                "window": 300,  # 5 minutes
                "severity": "high"
            },
            "unusual_file_access": {
                "paths": ["/etc", "/var/log", "/boot", "/root"],
                "severity": "medium"
            },
            "unusual_outbound_connections": {
                "threshold": 10,
                "window": 60,  # 1 minute
                "severity": "medium"
            },
            "system_file_changes": {
                "severity": "critical"
            }
        }
    
    def establish_baseline(self, traffic_data, period=3600):
        """Establish baseline network activity for anomaly detection"""
        if not traffic_data:
            self.logger.warning("No data provided to establish baseline")
            return False
            
        self.baseline = {
            'timestamp': time.time(),
            'period': period,
            'traffic_volume': sum(pkt.get('size', 0) for pkt in traffic_data) / len(traffic_data),
            'connection_rate': len(set(pkt.get('source_ip') for pkt in traffic_data)) / (period/3600),
            'error_rate': sum(1 for pkt in traffic_data if pkt.get('status', 200) >= 400) / len(traffic_data)
        }
        
        self.logger.info(f"Baseline established: {self.baseline}")
        return True
    
    def detect_anomalies(self, traffic_data, window=300):
        """Detect anomalies based on baseline comparison"""
        if not self.running:
            self.logger.warning("Cannot analyze - IDS not running")
            return []
            
        if not self.baseline:
            self.logger.warning("No baseline established for anomaly detection")
            return []
            
        anomalies = []
        
        # Calculate current metrics
        current = {
            'traffic_volume': sum(pkt.get('size', 0) for pkt in traffic_data) / len(traffic_data),
            'connection_rate': len(set(pkt.get('source_ip') for pkt in traffic_data)) / (window/3600),
            'error_rate': sum(1 for pkt in traffic_data if pkt.get('status', 200) >= 400) / len(traffic_data)
        }
        
        # Compare with baseline
        for metric, value in current.items():
            baseline_value = self.baseline[metric]
            threshold = self.anomaly_thresholds[metric]
            
            if value > baseline_value * threshold:
                anomalies.append({
                    'type': f'anomalous_{metric}',
                    'severity': 'medium',
                    'details': f'Current {metric}: {value}, Baseline: {baseline_value}, Threshold: {threshold}',
                    'timestamp': time.time()
                })
                
        return anomalies
    
    def monitor_authentication(self, auth_logs):
        """Monitor authentication logs for suspicious activity"""
        if not self.running:
            return []
            
        alerts = []
        auth_failures = defaultdict(list)
        
        # Process auth logs
        for log in auth_logs:
            if log.get('success') is False:
                user = log.get('user', 'unknown')
                source = log.get('source', 'unknown')
                timestamp = log.get('timestamp', time.time())
                
                # Track failures by user and source
                key = f"{user}:{source}"
                auth_failures[key].append(timestamp)
                
                # Check against threshold
                rule = self.detection_rules["repeated_auth_failures"]
                recent_failures = [t for t in auth_failures[key] 
                                  if timestamp - t <= rule["window"]]
                
                if len(recent_failures) >= rule["threshold"]:
                    alerts.append({
                        'type': 'repeated_auth_failures',
                        'severity': rule['severity'],
                        'details': f'Multiple auth failures for {user} from {source}',
                        'timestamp': timestamp
                    })
        
        return alerts
    
    def get_status(self):
        """Return the current IDS status"""
        return {
            "running": self.running,
            "rules_loaded": len(self.detection_rules),
            "alerts_generated": len(self.alerts),
            "baseline_established": bool(self.baseline)
        }