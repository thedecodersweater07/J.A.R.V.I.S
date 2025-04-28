# threat_analyzer.py
# Threat analysis system for the Jarvis security suite

import logging
import time
import re
import json
from collections import defaultdict

class ThreatAnalyzer:
    """Analyzes incoming data and events to identify potential security threats"""
    
    def __init__(self):
        self.logger = logging.getLogger("ThreatAnalyzer")
        self.running = False
        self.threat_definitions = self._load_threat_definitions()
        self.threat_patterns = self._compile_patterns()
        self.detection_stats = defaultdict(int)
        self.suspicious_ips = set()
        self.last_analysis = None
        self.logger.info("Threat Analyzer initialized")
    
    def start(self):
        """Start the threat analysis system"""
        self.running = True
        self.logger.info("Threat Analyzer started")
        return True
    
    def stop(self):
        """Stop the threat analysis system"""
        self.running = False
        self.logger.info("Threat Analyzer stopped")
        return True
        
    def _load_threat_definitions(self):
        """Load threat definitions from configuration"""
        # In a real system, this would load from a file or database
        # For this example, we'll define some basic threats
        return {
            "sql_injection": {
                "patterns": [
                    r"(?i)('|\"|;)\s*(OR|AND)\s*('|\"|;)",
                    r"(?i)UNION\s+SELECT",
                    r"(?i)INSERT\s+INTO.*VALUES"
                ],
                "severity": "high"
            },
            "xss_attack": {
                "patterns": [
                    r"(?i)<script.*?>.*?</script>",
                    r"(?i)javascript:",
                    r"(?i)on(load|click|mouseover|error)="
                ],
                "severity": "high"
            },
            "path_traversal": {
                "patterns": [
                    r"(?:\.\./|\.\.\\|\.\.$)",
                    r"(?:/etc/(?:passwd|shadow|master\.passwd))",
                    r"(?:C:\\windows\\system32)"
                ],
                "severity": "critical"
            },
            "port_scan": {
                "patterns": [],  # Detected through connection pattern analysis, not string patterns
                "severity": "medium"
            },
            "brute_force": {
                "patterns": [],  # Detected through login attempt frequency
                "severity": "high"
            }
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        compiled_patterns = {}
        for threat_name, threat_info in self.threat_definitions.items():
            compiled_patterns[threat_name] = [re.compile(pattern) for pattern in threat_info["patterns"]]
        return compiled_patterns
    
    def analyze_network_traffic(self, traffic_data):
        """Analyze network traffic for potential threats"""
        if not self.running:
            self.logger.warning("Cannot analyze traffic - analyzer not running")
            return None
            
        self.logger.debug(f"Analyzing network traffic: {len(traffic_data)} packets")
        self.last_analysis = time.time()
        
        detected_threats = []
        
        # Check for port scanning behavior
        port_scan_threats = self._detect_port_scans(traffic_data)
        if port_scan_threats:
            detected_threats.extend(port_scan_threats)
        
        # Check for brute force attempts
        brute_force_threats = self._detect_brute_force(traffic_data)
        if brute_force_threats:
            detected_threats.extend(brute_force_threats)
        
        # Check packet payloads for known attack patterns
        pattern_threats = self._analyze_packet_payloads(traffic_data)
        if pattern_threats:
            detected_threats.extend(pattern_threats)
        
        # Update detection statistics
        for threat in detected_threats:
            self.detection_stats[threat['type']] += 1
            if 'source_ip' in threat:
                self.suspicious_ips.add(threat['source_ip'])
        
        return detected_threats
    
    def _detect_port_scans(self, traffic_data):
        """Detect potential port scanning activities"""
        port_scan_threats = []
        ip_port_counts = defaultdict(set)
        
        # Count unique ports accessed by each IP
        for packet in traffic_data:
            if 'source_ip' in packet and 'dest_port' in packet:
                ip_port_counts[packet['source_ip']].add(packet['dest_port'])
        
        # Check for IPs accessing many ports in a short time
        for ip, ports in ip_port_counts.items():
            if len(ports) > 15:  # Threshold for suspicious port access
                port_scan_threats.append({
                    'type': 'port_scan',
                    'severity': self.threat_definitions['port_scan']['severity'],
                    'details': f'IP {ip} accessed {len(ports)} different ports',
                    'source_ip': ip
                })
                
        return port_scan_threats
    
    def _detect_brute_force(self, traffic_data):
        """Detect potential brute force login attempts"""
        brute_force_threats = []
        login_attempts = defaultdict(int)
        
        # Count login attempts by IP
        for packet in traffic_data:
            if packet.get('protocol') == 'HTTP' and packet.get('path', '').endswith('/login'):
                if packet.get('method') == 'POST':
                    login_attempts[packet['source_ip']] += 1
        
        # Check for excessive login attempts
        for ip, count in login_attempts.items():
            if count > 10:  # Threshold for suspicious login attempts
                brute_force_threats.append({
                    'type': 'brute_force',
                    'severity': self.threat_definitions['brute_force']['severity'],
                    'details': f'IP {ip} made {count} login attempts',
                    'source_ip': ip
                })
                
        return brute_force_threats
    
    def _analyze_packet_payloads(self, traffic_data):
        """Analyze packet payloads for attack patterns"""
        payload_threats = []
        
        for packet in traffic_data:
            if 'payload' not in packet:
                continue
                
            payload = packet['payload']
            source_ip = packet.get('source_ip', 'unknown')
            
            for threat_name, patterns in self.threat_patterns.items():
                if not patterns:  # Skip threats without patterns
                    continue
                    
                for pattern in patterns:
                    if pattern.search(payload):
                        payload_threats.append({
                            'type': threat_name,
                            'severity': self.threat_definitions[threat_name]['severity'],
                            'details': f'{threat_name} pattern detected in payload',
                            'source_ip': source_ip,
                            'pattern_matched': pattern.pattern
                        })
                        break  # Stop checking patterns for this threat
        
        return payload_threats
    
    def analyze_log_entry(self, log_entry):
        """Analyze a log entry for potential threats"""
        if not self.running:
            self.logger.warning("Cannot analyze log entry - analyzer not running")
            return None
            
        self.last_analysis = time.time()
        detected_threats = []
        
        # Convert log entry to string if it's not already
        if isinstance(log_entry, dict):
            log_text = json.dumps(log_entry)
        else:
            log_text = str(log_entry)
        
        # Check log against threat patterns
        for threat_name, patterns in self.threat_patterns.items():
            if not patterns:
                continue
                
            for pattern in patterns:
                if pattern.search(log_text):
                    source_ip = log_entry.get('source_ip', None) if isinstance(log_entry, dict) else None
                    
                    threat = {
                        'type': threat_name,
                        'severity': self.threat_definitions[threat_name]['severity'],
                        'details': f'{threat_name} pattern detected in log entry'
                    }
                    
                    if source_ip:
                        threat['source_ip'] = source_ip
                        self.suspicious_ips.add(source_ip)
                        
                    detected_threats.append(threat)
                    self.detection_stats[threat_name] += 1
                    break
        
        return detected_threats if detected_threats else None
    
    def get_threat_summary(self):
        """Get a summary of detected threats"""
        return {
            "total_detections": sum(self.detection_stats.values()),
            "detection_by_type": dict(self.detection_stats),
            "suspicious_ips": list(self.suspicious_ips),
            "last_analysis": self.last_analysis
        }
    
    def get_status(self):
        """Return the current status of the analyzer"""
        return {
            "running": self.running,
            "threat_types_monitored": len(self.threat_definitions),
            "total_detections": sum(self.detection_stats.values()),
            "suspicious_ips_count": len(self.suspicious_ips)
        }