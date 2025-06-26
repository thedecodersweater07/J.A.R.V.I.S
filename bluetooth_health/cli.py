"""
Command-line interface for Bluetooth health monitoring.
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

# Import BluetoothHealthMonitor with proper error handling
try:
    # Try to import from the installed package first
    from bluetooth_health.core import BluetoothHealthMonitor
except ImportError:
    # If that fails, try a relative import
    try:
        from .core import BluetoothHealthMonitor
    except ImportError:
        # If both fail, try to import directly from the core module
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from bluetooth_health.core import BluetoothHealthMonitor
        except ImportError as e:
            raise ImportError(
                "Could not import BluetoothHealthMonitor. "
                "Make sure you're running from the correct directory and the package is properly installed.\n"
                f"Error: {e}"
            ) from e

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def list_devices():
    """List available Bluetooth devices."""
    monitor = BluetoothHealthMonitor()
    devices = await monitor.discover_devices()
    
    if not devices:
        print("No Bluetooth devices found.")
        return
        
    print("\nAvailable Bluetooth devices:")
    print("-" * 40)
    for name, address in devices.items():
        print(f"Name: {name}")
        print(f"Address: {address}")
        print("-" * 40)

async def monitor_heart_rate(device_name: Optional[str] = None, device_address: Optional[str] = None):
    """Monitor heart rate from a Bluetooth device."""
    monitor = BluetoothHealthMonitor(device_name, device_address)
    
    def on_heart_rate(rate: int):
        print(f"\rHeart rate: {rate} BPM", end="", flush=True)
    
    try:
        device_info = f"device {device_name}" if device_name else f"device at {device_address}" if device_address else "a device"
        print(f"Connecting to {device_info}...")
        if not await monitor.connect():
            return
            
        print("Connected! Starting heart rate monitoring... (Press Ctrl+C to stop)")
        await monitor.start_heart_rate_monitoring(on_heart_rate)
        
        # Keep the connection alive
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        print("\nStopping...")
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        await monitor.stop_heart_rate_monitoring()
        await monitor.disconnect()

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Bluetooth Health Monitor")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List devices command
    list_parser = subparsers.add_parser("list", help="List available Bluetooth devices")
    list_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor heart rate")
    monitor_parser.add_argument("-n", "--name", help="Name of the device to connect to")
    monitor_parser.add_argument("-a", "--address", help="MAC address of the device to connect to")
    monitor_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)
    
    try:
        if args.command == "list":
            asyncio.run(list_devices())
        elif args.command == "monitor":
            if not args.name and not args.address:
                print("Error: You must specify either --name or --address")
                monitor_parser.print_help()
                return
            asyncio.run(monitor_heart_rate(args.name, args.address))
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
