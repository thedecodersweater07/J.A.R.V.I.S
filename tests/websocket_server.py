import asyncio
import websockets
import json
import logging
import signal
import sys
import os
from pathlib import Path

# Zorg ervoor dat de huidige map in het pad staat
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configureer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('JarvisWSServer')

# Importeer de eenvoudige Jarvis implementatie
from simple_jarvis import SimpleJarvis

class WebSocketServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.server = None
        self.jarvis = SimpleJarvis()
    
    async def register(self, websocket):
        self.clients.add(websocket)
        logger.info(f'Nieuwe client verbonden. Totaal: {len(self.clients)}')
        
        # Stuur een welkomstbericht
        try:
            await websocket.send(json.dumps({
                'type': 'system',
                'message': 'Verbonden met Jarvis AI. Stel je vraag!'
            }))
        except Exception as e:
            logger.error(f'Fout bij verzenden welkomstbericht: {e}')
    
    async def unregister(self, websocket):
        if websocket in self.clients:
            self.clients.remove(websocket)
            logger.info(f'Client verbroken. Resterende clients: {len(self.clients)}')
    
    async def handle_client(self, websocket, path):
        await self.register(websocket)
        
        try:
            async for message in websocket:
                try:
                    # Parse het binnenkomende bericht
                    try:
                        data = json.loads(message)
                        message_text = data.get('message', '').strip()
                        logger.info(f"Ontvangen bericht: {message_text}")
                    except json.JSONDecodeError:
                        error_msg = 'Ongeldig JSON-formaat ontvangen'
                        logger.error(error_msg)
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': error_msg
                        }))
                        continue
                    
                    if not message_text:
                        error_msg = 'Leeg bericht ontvangen'
                        logger.warning(error_msg)
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': error_msg
                        }))
                        continue
                    
                    # Verwerk het bericht met de message handler
                    try:
                        response = await self.jarvis.handle_message(message_text)
                        logger.info(f"Verstuur antwoord: {response[:100]}..." if len(str(response)) > 100 else f"Verstuur antwoord: {response}")
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'message': response
                        }))
                    except Exception as e:
                        error_msg = f'Fout bij verwerken bericht: {str(e)}'
                        logger.error(error_msg, exc_info=True)
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': error_msg
                        }))
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.info('Verbinding gesloten door client')
                    break
                except Exception as e:
                    error_msg = f'Onverwachte fout: {str(e)}'
                    logger.error(error_msg, exc_info=True)
                    try:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Interne serverfout'
                        }))
                    except:
                        pass
                    break
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10
        )
        logger.info(f'WebSocket server gestart op ws://{self.host}:{self.port}')
        return self.server
    
    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info('WebSocket server gestopt')

async def main():
    # Maak een nieuwe WebSocket server
    server = WebSocketServer()
    
    # Registreer signaalhandlers voor nette afsluiting
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(shutdown(s, server, loop))
        )
    
    try:
        # Start de server
        await server.start()
        
        # Blijf actief tot de server wordt gestopt
        await asyncio.Future()
        
    except asyncio.CancelledError:
        logger.info('Server wordt netjes afgesloten...')
    except Exception as e:
        logger.exception('Onverwachte fout opgetreden:')
    finally:
        # Zorg voor een nette afsluiting
        await server.stop()

async def shutdown(signal, server, loop):
    """Netjes afsluiten bij het ontvangen van een signaal"""
    logger.info(f'\nOntvangen signaal {signal.name}...')
    await server.stop()
    loop.stop()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('Server gestopt door gebruiker')
    except Exception as e:
        logger.exception('Fout opgetreden:')
        sys.exit(1)
