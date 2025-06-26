# Bluetooth Health Monitor

A Python module for monitoring health data from Bluetooth devices like smartwatches and fitness trackers.

## Features

- Discover nearby Bluetooth LE devices
- Connect to health devices (heart rate monitors, etc.)
- Real-time heart rate monitoring
- Command-line interface
- REST API for integration with web applications
- WebSocket support for real-time updates

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Command Line Usage

### List available Bluetooth devices:
```
python -m bluetooth_health.cli list
```

### Monitor heart rate from a device:
```
# By device name
python -m bluetooth_health.cli monitor --name "Your Device Name"

# Or by MAC address
python -m bluetooth_health.cli monitor --address "00:11:22:33:44:55"
```

## API Usage

### Start the API server:
```
python -m bluetooth_health.api
```

### Available API Endpoints:

- `GET /api/devices` - List available Bluetooth devices
- `POST /api/connect` - Connect to a device
- `POST /api/start-monitoring` - Start monitoring health data
- `POST /api/stop-monitoring` - Stop monitoring
- `POST /api/disconnect` - Disconnect from the device
- `GET /docs` - Interactive API documentation (Swagger UI)

### WebSocket Endpoint:
- `ws://localhost:8000/ws` - Real-time health data updates

## Integration with Frontend

1. Install the required dependencies in your frontend project:
   ```
   npm install --save socket.io-client
   ```

2. Connect to the WebSocket endpoint and handle incoming data:
   ```javascript
   import { io } from 'socket.io-client';

   const socket = io('ws://localhost:8000');
   
   socket.on('connect', () => {
     console.log('Connected to WebSocket server');
   });
   
   socket.on('heart_rate', (data) => {
     console.log('Heart rate:', data);
     // Update your UI with the new heart rate
   });
   ```

## License

MIT
