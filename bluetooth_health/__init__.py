"""
Bluetooth Health Monitor Module

This module provides functionality to connect to Bluetooth devices (phones, smartwatches)
to monitor health metrics like heart rate, time, and other health-related data.
"""

# Version of the package
__version__ = "0.1.0"

# Import core functionality
try:
    # Try absolute import first (when installed as a package)
    from bluetooth_health.core import BluetoothHealthMonitor
except ImportError:
    try:
        # Try relative import (when running directly)
        from .core import BluetoothHealthMonitor
    except ImportError as e:
        raise ImportError(
            "Could not import BluetoothHealthMonitor. "
            "Make sure the package is properly installed."
        ) from e

# Lazy loading for API and CLI to avoid circular imports
_api_app = None
_cli_main = None

def _get_api_app():
    global _api_app
    if _api_app is None:
        try:
            from .api import app as api_app
            _api_app = api_app
        except ImportError as e:
            raise ImportError("Could not import API app. Make sure all dependencies are installed.") from e
    return _api_app

def _get_cli_main():
    global _cli_main
    if _cli_main is None:
        try:
            from .cli import main as cli_main
            _cli_main = cli_main
        except ImportError as e:
            raise ImportError("Could not import CLI main. Make sure all dependencies are installed.") from e
    return _cli_main

# Create properties for lazy loading
api_app = property(_get_api_app)
cli_main = property(_get_cli_main)

__all__ = ['BluetoothHealthMonitor', 'api_app', 'cli_main']
