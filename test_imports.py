"""Test script to verify imports are working correctly."""

try:
    from bluetooth_health import BluetoothHealthMonitor
    print("Successfully imported BluetoothHealthMonitor")
    
    # Test creating an instance
    monitor = BluetoothHealthMonitor()
    print("Successfully created BluetoothHealthMonitor instance")
    
    # Test importing other modules
    from bluetooth_health import api_app, cli_main
    print("Successfully imported api_app and cli_main")
    
    print("\nAll imports successful!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nTroubleshooting:")
    print("1. Make sure you've installed the package in development mode:")
    print("   pip install -e .")
    print("2. Check that you're running this from the root directory of the project")
    print("3. Verify that the bluetooth_health directory contains an __init__.py file")
