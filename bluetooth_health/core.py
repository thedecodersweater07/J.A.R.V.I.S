from __future__ import annotations

"""
Core Bluetooth functionality for health monitoring.
"""
import asyncio
import logging
import sys
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, cast, List

# Type variable for generic types
T = TypeVar('T')

# Type variable for generic types
T = TypeVar('T')

# Try to import Bleak with proper error handling
try:
    from bleak import BleakScanner, BleakClient
    BLEAK_AVAILABLE = True
except ImportError:
    # Define dummy implementations if Bleak is not available
    class DummyBleakScanner:
        @staticmethod
        async def discover() -> List[Any]:
            raise RuntimeError("Bleak is not installed. Please install with: pip install bleak")
    
    class DummyBleakClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Bleak is not installed. Please install with: pip install bleak")
    
    BleakScanner = DummyBleakScanner  # type: ignore
    BleakClient = DummyBleakClient  # type: ignore
    BLEAK_AVAILABLE = False

# Standard Bluetooth UUIDs for health services
HEART_RATE_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HEART_RATE_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"
DEVICE_INFORMATION_SERVICE = "0000180a-0000-1000-8000-00805f9b34fb"

class BluetoothHealthMonitor:
    def __init__(self, device_name: Optional[str] = None, device_address: Optional[str] = None):
        """
        Initialize the Bluetooth health monitor.
        
        Args:
            device_name: Optional name of the device to connect to
            device_address: Optional MAC address of the device to connect to
        """
        self.device_name = device_name
        self.device_address = device_address
        self.client: Optional[Any] = None  # Using Any to avoid type issues with Bleak
        self.connected = False
        self.heart_rate_callback: Optional[Callable[[int], None]] = None
        self.logger = logging.getLogger(__name__)

    async def discover_devices(self) -> Dict[str, str]:
        """Scan for nearby Bluetooth LE devices."""
        self.logger.info("Scanning for Bluetooth LE devices...")
        try:
            devices = await BleakScanner.discover()
            return {str(d.name) or "Unknown": str(d.address) for d in devices}  # type: ignore
        except Exception as e:
            self.logger.error(f"Error discovering devices: {e}")
            return {}

    async def connect(self) -> bool:
        """Connect to the specified Bluetooth device."""
        if self.device_address is None:
            devices = await self.discover_devices()
            if self.device_name and self.device_name in devices:
                self.device_address = devices[self.device_name]
            else:
                self.logger.error("No device address specified and couldn't find device by name")
                return False

        if not self.device_address:
            self.logger.error("No device address available to connect")
            return False

        try:
            # Create a new client instance
            self.client = BleakClient(str(self.device_address))
            if not self.client:
                self.logger.error("Failed to create Bluetooth client")
                return False
                
            # Connect to the device
            await self.client.connect()
            self.connected = True
            self.logger.info(f"Connected to {self.device_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.device_address}: {e}")
            self.connected = False
            self.client = None
            return False

    async def disconnect(self):
        """Disconnect from the Bluetooth device."""
        if self.client and self.connected:
            await self.client.disconnect()
            self.connected = False
            self.logger.info("Disconnected from device")

    def _heart_rate_callback(self, sender: str, data: bytearray):
        """Process heart rate measurement data."""
        # Heart Rate Measurement format:
        # Flags (1 byte) + Heart Rate Value (8 or 16 bits) + optional fields
        flags = data[0]
        is_hr_in_16bit = bool(flags & 0x01)
        
        if is_hr_in_16bit:
            heart_rate = int.from_bytes(data[1:3], byteorder='little')
        else:
            heart_rate = int(data[1])
            
        if self.heart_rate_callback:
            self.heart_rate_callback(heart_rate)

    async def start_heart_rate_monitoring(self, callback: Callable[[int], None]):
        """Start monitoring heart rate data.
        
        Args:
            callback: Function to call when new heart rate data is received
        """
        if not self.connected or not self.client:
            raise RuntimeError("Not connected to a device")
            
        self.heart_rate_callback = callback
        await self.client.start_notify(HEART_RATE_MEASUREMENT, self._heart_rate_callback)
        self.logger.info("Started heart rate monitoring")

    async def stop_heart_rate_monitoring(self):
        """Stop monitoring heart rate data."""
        if self.connected and self.client:
            await self.client.stop_notify(HEART_RATE_MEASUREMENT)
            self.heart_rate_callback = None
            self.logger.info("Stopped heart rate monitoring")
