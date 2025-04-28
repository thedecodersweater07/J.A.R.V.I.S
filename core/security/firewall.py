# firewall.py
# Network firewall system for the Jarvis security suite

import logging
import time
import re
import ipaddress
from collections import defaultdict

class Firewall:
    """Implements a network firewall for Jarvis"""
    
    def __init__(self):
        self.logger = logging.getLogger("Firewall")
        self.running = False
        self.rules = []
        self.blocked_ips = set()
        self.temporary_blocks = {}  # IP: expiration_time
        self.connection_limits = {}  # IP: {count: int, last_reset: timestamp}
        self.default_policy = "allow"  # "allow" or "deny"
        self.log_dropped = True
        self.logger.info("Firewall initialized")
    
    def start(self):
        """Start the firewall"""
        self.running = True
        self.logger.info("Firewall started")
        return True
    
    def stop(self):
        """Stop the firewall"""
        self.running = False
        self.logger.info("Firewall stopped")
        return True
    
    def add_rule(self, rule):
        """Add a firewall rule"""
        if not self._validate_rule(rule):
            self.logger.error(f"Invalid rule format: {rule}")
            return False
            
        self.rules.append(rule)
        self.logger.info(f"Added firewall rule: {rule['name']}")
        return True
    
    def remove_rule(self, rule_name):
        """Remove a firewall rule by name"""
        initial_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule['name'] != rule_name]
        
        if len(self.rules) < initial_count:
            self.logger.info(f"Removed firewall rule: {rule_name}")
            return True
        else:
            self.logger.warning(f"Rule not found: {rule_name}")
            return False
    
    def _validate_rule(self, rule):
        """Validate a firewall rule format"""
        required_fields = ['name', 'action']
        
        # Check required fields
        for field in required_fields:
            if field not in rule:
                return False
        
        # Validate action
        if rule['action'] not in ['allow', 'deny', 'log']:
            return False
            
        # Validate IP or network if present
        if 'source_ip' in rule:
            try:
                # Check if it's a CIDR notation or single IP
                ipaddress.ip_network(rule['source_ip'], strict=False)
            except ValueError:
                return False
        
        # Validate port range if present
        if 'port' in rule:
            try:
                if '-' in str(rule['port']):
                    start, end = map(int, rule['port'].split('-'))
                    if not (0 <= start <= end <= 65535):
                        return False
                else:
                    port = int(rule['port'])
                    if not (0 <= port <= 65535):
                        return False
            except (ValueError, AttributeError):
                return False
        
        return True
    
    def evaluate_packet(self, packet):
        """Evaluate a packet against firewall rules
        
        Returns:
            bool: True if packet is allowed, False if blocked
        """
        if not self.running:
            self.logger.warning("Cannot evaluate packet - firewall not running")
            return True  # Allow by default if firewall is not running
        
        # Get source IP from packet
        source_ip = packet.get('source_ip')
        
        # Check if IP is permanently blocked
        if source_ip in self.blocked_ips:
            if self.log_dropped:
                self.logger.info(f"Blocked packet from permanently blocked IP: {source_ip}")
            return False
        
        # Check if IP is temporarily blocked
        if source_ip in self.temporary_blocks:
            if time.time() < self.temporary_blocks[source_ip]:
                if self.log_dropped:
                    self.logger.info(f"Blocked packet from temporarily blocked IP: {source_ip}")
                return False
            else:
                # Block expired, remove it
                del self.temporary_blocks[source_ip]
        
        # Check connection limits
        if source_ip in self.connection_limits:
            limit_data = self.connection_limits[source_ip]
            
            # Reset counter if needed
            if time.time() - limit_data['last_reset'] > 60:  # 1-minute window
                limit_data['count'] = 1
                limit_data['last_reset'] = time.time()
            else:
                limit_data['count'] += 1
            
            # Check if limit exceeded
            if limit_data['count'] > limit_data['limit']:
                if self.log_dropped:
                    self.logger.warning(f"Connection limit exceeded for IP: {source_ip}")
                return False
        
        # Evaluate against firewall rules
        for rule in self.rules:
            if self._packet_matches_rule(packet, rule):
                if rule['action'] == 'allow':
                    return True
                elif rule['action'] == 'deny':
                    if self.log_dropped:
                        self.logger.info(f"Packet dropped due to rule: {rule['name']}")
                    return False
                elif rule['action'] == 'log':
                    self.logger.info(f"Logged packet matching rule: {rule['name']}")
                    # Continue checking other rules
        
        # If no rules matched, apply default policy
        return self.default_policy == "allow"
    
    def _packet_matches_rule(self, packet, rule):
        """Check if a packet matches a firewall rule"""
        # Check source IP
        if 'source_ip' in rule:
            if 'source_ip' not in packet:
                return False
                
            try:
                rule_network = ipaddress.ip_network(rule['source_ip'], strict=False)
                packet_ip = ipaddress.ip_address(packet['source_ip'])
                if packet_ip not in rule_network:
                    return False
            except (ValueError, TypeError):
                return False
        
        # Check destination IP
        if 'dest_ip' in rule:
            if 'dest_ip' not in packet:
                return False
                
            try:
                rule_network = ipaddress.ip_network(rule['dest_ip'], strict=False)
                packet_ip = ipaddress.ip_address(packet['dest_ip'])
                if packet_ip not in rule_network:
                    return False
            except (ValueError, TypeError):
                return False
        
        # Check protocol
        if 'protocol' in rule and rule['protocol'] != packet.get('protocol'):
            return False
        
        # Check port
        if 'port' in rule:
            packet_port = packet.get('dest_port')
            if not packet_port:
                return False
                
            try:
                if '-' in str(rule['port']):
                    start, end = map(int, rule['port'].split('-'))
                    if not (start <= int(packet_port) <= end):
                        return False
                else:
                    if int(packet_port) != int(rule['port']):
                        return False
            except (ValueError, TypeError):
                return False
        
        # If we got here, packet matches rule
        return True
    
    def block_ip(self, ip, duration=None):
        """Block an IP address
        
        Args:
            ip (str): IP address to block
            duration (int, optional): Duration in seconds. If None, block is permanent.
        """
        try:
            # Validate IP address
            ipaddress.ip_address(ip)
            
            if duration is None:
                # Permanent block
                self.blocked_ips.add(ip)
                self.logger.warning(f"Permanently blocked IP: {ip}")
            else:
                # Temporary block
                expiration = time.time() + duration
                self.temporary_blocks[ip] = expiration
                self.logger.warning(f"Temporarily blocked IP: {ip} for {duration} seconds")
            
            return True
        except ValueError:
            self.logger.error(f"Invalid IP address: {ip}")
            return False
    
    def unblock_ip(self, ip):
        """Unblock an IP address"""
        removed = False
        
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            removed = True
            
        if ip in self.temporary_blocks:
            del self.temporary_blocks[ip]
            removed = True
            
        if removed:
            self.logger.info(f"Unblocked IP: {ip}")
            return True
        else:
            self.logger.warning(f"IP not found in block lists: {ip}")
            return False
    
    def set_connection_limit(self, ip, limit):
        """Set connection limit for an IP
        
        Args:
            ip (str): IP address
            limit (int): Maximum connections per minute
        """
        try:
            ipaddress.ip_address(ip)
            self.connection_limits[ip] = {
                'limit': limit,
                'count': 0,
                'last_reset': time.time()
            }
            self.logger.info(f"Set connection limit for {ip}: {limit} per minute")
            return True
        except ValueError:
            self.logger.error(f"Invalid IP address: {ip}")
            return False
    
    def set_default_policy(self, policy):
        """Set the default policy for packets that don't match any rules"""
        if policy in ['allow', 'deny']:
            self.default_policy = policy
            self.logger.info(f"Default policy set to: {policy}")
            return True
        else:
            self.logger.error(f"Invalid policy: {policy}")
            return False
    
    def get_status(self):
        """Return the current status of the firewall"""
        # Clean up expired temporary blocks
        current_time = time.time()
        expired = [ip for ip, expiry in self.temporary_blocks.items() if current_time > expiry]
        for ip in expired:
            del self.temporary_blocks[ip]
            
        return {
            "running": self.running,
            "rules_count": len(self.rules),
            "default_policy": self.default_policy,
            "permanent_blocks": len(self.blocked_ips),
            "temporary_blocks": len(self.temporary_blocks),
            "connection_limits": len(self.connection_limits)
        }