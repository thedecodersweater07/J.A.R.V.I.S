import asyncio
import websockets

async def test_websocket():
    try:
        async with websockets.connect('ws://localhost:18866') as websocket:
            print("Verbinding gemaakt met WebSocket server")
            await websocket.send('test')
            response = await websocket.recv()
            print(f"Ontvangen: {response}")
    except Exception as e:
        print(f"Fout: {e}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_websocket())
