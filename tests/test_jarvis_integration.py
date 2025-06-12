import asyncio
import websockets
import json
import logging
import signal
import os
import sys
import socket
import argparse
from typing import Optional, Tuple

# Zorg ervoor dat de project root in het pad staat
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Windows-specifieke imports
try:
    if os.name == 'nt':
        import win32api
        import win32con
except ImportError:
    if os.name == 'nt':
        logging.warning("pywin32 module niet gevonden, beperkte signaalondersteuning op Windows")

import http.server
import socketserver
import threading
import time
import uuid
import random
from pathlib import Path
from http import HTTPStatus
from datetime import datetime as dt
from urllib.parse import unquote
from typing import Set, Optional, Dict, Any, Callable, Awaitable

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configureer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis_server.log')
    ]
)
logger = logging.getLogger('JarvisServer')

# Gebruik de eenvoudige implementatie in plaats van de volledige JarvisAI
logger.info('Gebruik eenvoudige implementatie van Jarvis')
class JarvisAI:
    async def process_message(self, message):
        responses = [
            f"Ik heb je bericht ontvangen: {message}",
            f"Interessant dat je zegt: {message}",
            f"Ik verwerk je verzoek over: {message}",
            f"Bedankt voor je bericht: {message}"
        ]
        import random
        return random.choice(responses)

# Importeer de eenvoudige Jarvis implementatie
try:
    from simple_jarvis import SimpleJarvis
except ImportError:
    # Als simple_jarvis niet beschikbaar is, maak dan een eenvoudige vervanging
    class SimpleJarvis:
        def __init__(self):
            self.logger = logging.getLogger('SimpleJarvis')
            self.logger.info('SimpleJarvis geïnitialiseerd')
        
        async def handle_message(self, message):
            # Eenvoudige patroonherkenning voor verschillende soorten berichten
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['hallo', 'hoi', 'hey', 'hoi daar']):
                return random.choice([
                    "Hallo! Hoe kan ik je vandaag helpen?",
                    "Hoi daar! Wat kan ik voor je doen?",
                    "Dag! Waar kan ik je mee assisteren?"
                ])
                
            elif any(word in message_lower for word in ['hoe gaat het', 'hoe is het', 'alles goed']):
                return random.choice([
                    "Met mij gaat het goed, bedankt! En met jou?",
                    "Alles gaat prima hier! Hoe gaat het met jou?",
                    "Ik functioneer optimaal. Kan ik je ergens mee helpen?"
                ])
                
            elif any(word in message_lower for word in ['wat kan je', 'wat doe je', 'help']):
                return "Ik kan je helpen met eenvoudige gesprekken. Je kunt me alles vragen!"
                
            elif 'dank' in message_lower or 'bedankt' in message_lower:
                return "Graag gedaan! Is er nog iets anders waar ik je mee kan helpen?"
                
            elif any(word in message_lower for word in ['stop', 'doei', 'tot ziens']):
                return "Tot ziens! Laat het me weten als je nog hulp nodig hebt."
                
            # Standaard antwoord als er geen specifiek patroon wordt herkend
            return random.choice([
                f"Interessant dat je zegt: '{message}'. Kun je daar meer over vertellen?",
                "Dat is een goede vraag. Laat me even nadenken...",
                "Ik begrijp wat je bedoelt. Kun je iets meer details geven?",
                f"Bedankt voor je bericht over '{message}'. Wat zou je hier nog meer over willen weten?",
                "Interessant punt! Heb je hier specifieke vragen over?"
            ])

class WebSocketServer:
    """WebSocket server met verbeterde foutafhandeling en herstelmechanismen."""
    
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.original_port = port  # Bewaar de originele poort voor foutmeldingen
        self.clients = set()
        self.server = None
        self.jarvis = SimpleJarvis()
        self.running = False
        self.start_attempts = 0
        self.max_start_attempts = 3
    
    async def register(self, websocket):
        """Registreer een nieuwe WebSocket client."""
        self.clients.add(websocket)
        client_id = id(websocket)
        logger.info(f'Nieuwe client verbonden (ID: {client_id}). Totaal: {len(self.clients)}')
        
        # Stuur een welkomstbericht
        try:
            await websocket.send(json.dumps({
                'type': 'system',
                'message': 'Verbonden met Jarvis AI. Stel je vraag!',
                'timestamp': dt.now().isoformat(),
                'client_id': client_id
            }))
            return client_id
        except Exception as e:
            logger.error(f'Fout bij verzenden welkomstbericht: {e}')
            return None
    
    async def unregister(self, websocket):
        """Verwijder een WebSocket client bij het verbreken van de verbinding."""
        if websocket in self.clients:
            client_id = id(websocket)
            self.clients.remove(websocket)
            logger.info(f'Client verbroken (ID: {client_id}). Resterende clients: {len(self.clients)}')
    
    async def broadcast(self, message, exclude=None):
        """Verstuur een bericht naar alle verbonden clients."""
        if not self.clients:
            return
            
        if exclude:
            clients = [client for client in self.clients if client != exclude]
        else:
            clients = list(self.clients)
            
        if not clients:
            return
            
        message_json = json.dumps(message) if not isinstance(message, str) else message
        
        await asyncio.gather(
            *[client.send(message_json) for client in clients],
            return_exceptions=True
        )
    
    async def handle_client(self, websocket, path):
        """Beheer een individuele client verbinding."""
        client_id = await self.register(websocket)
        if client_id is None:
            return
            
        try:
            async for message in websocket:
                try:
                    # Parse het binnenkomende bericht
                    try:
                        data = json.loads(message)
                        message_text = data.get('message', '').strip()
                        request_id = data.get('request_id', str(uuid.uuid4()))
                        logger.info(f"Ontvangen van client {client_id}: {message_text}")
                    except json.JSONDecodeError:
                        error_msg = 'Ongeldig JSON-formaat ontvangen'
                        logger.error(error_msg)
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': error_msg,
                            'request_id': request_id,
                            'timestamp': dt.now().isoformat()
                        }))
                        continue
                    
                    if not message_text:
                        error_msg = 'Leeg bericht ontvangen'
                        logger.warning(error_msg)
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': error_msg,
                            'request_id': request_id,
                            'timestamp': dt.now().isoformat()
                        }))
                        continue
                    
                    # Verwerk het bericht met de message handler
                    try:
                        response = await self.jarvis.handle_message(message_text)
                        logger.info(f"Verstuur antwoord naar client {client_id}: {response[:100]}..." if len(str(response)) > 100 else f"Verstuur antwoord: {response}")
                        
                        await websocket.send(json.dumps({
                            'type': 'response',
                            'message': response,
                            'request_id': request_id,
                            'timestamp': dt.now().isoformat()
                        }))
                        
                    except Exception as e:
                        error_msg = f'Fout bij verwerken bericht: {str(e)}'
                        logger.error(error_msg, exc_info=True)
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': error_msg,
                            'request_id': request_id,
                            'timestamp': dt.now().isoformat()
                        }))
                        
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f'Verbinding gesloten door client {client_id}')
                    break
                except Exception as e:
                    error_msg = f'Onverwachte fout: {str(e)}'
                    logger.error(error_msg, exc_info=True)
                    try:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'Interne serverfout',
                            'request_id': request_id,
                            'timestamp': dt.now().isoformat()
                        }))
                    except:
                        pass
                    break
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """Start de WebSocket server."""
        if self.running:
            logger.warning('WebSocket server is al gestart')
            return
            
        self.start_attempts += 1
        
        try:
            # Controleer of de poort beschikbaar is
            if is_port_in_use(self.port, self.host):
                # Probeer een alternatieve poort te vinden
                new_port = self.find_available_port(self.port + 1)
                if new_port is not None:
                    logger.warning(f'Poort {self.port} is al in gebruik, probeer poort {new_port}...')
                    self.port = new_port
                else:
                    raise OSError(
                        f'Kan geen beschikbare poort vinden in het bereik {self.port}-{self.port + 10}.'
                    )
            
            # Maak een nieuw event loop object voor deze thread
            loop = asyncio.get_event_loop()
            
            # Start de WebSocket server
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=30,  # Stuur elke 30 seconden een ping
                ping_timeout=10,    # Wacht 10 seconden op een pong
                close_timeout=5,    # Wacht 5 seconden bij het sluiten
                reuse_address=True  # Sta hergebruik van het adres toe
            )
            
            self.running = True
            logger.info(f'✅ WebSocket server gestart op ws://{self.host}:{self.port}')
            return True
            
        except OSError as e:
            if hasattr(e, 'winerror') and e.winerror == 10048:  # WSAEADDRINUSE
                logger.error(f'Poort {self.port} is al in gebruik. Probeer een andere poort of sluit de vorige instantie af.')
            else:
                logger.error(f'Fout bij starten WebSocket server: {e}')
            self.running = False
            return False
            
        except Exception as e:
            logger.error(f'Onverwachte fout bij starten WebSocket server: {e}', exc_info=True)
            self.running = False
            return False
    
    async def stop(self):
        """Stop de WebSocket server."""
        if not self.running or not self.server:
            return
            
        logger.info('Stoppen van WebSocket server...')
        self.running = False
        
        # Sluit alle actieve verbindingen
        if self.clients:
            await self.broadcast({
                'type': 'system',
                'message': 'Server wordt afgesloten. Verbinding wordt verbroken.',
                'timestamp': dt.now().isoformat()
            })
            
            # Geef clients even de tijd om het bericht te ontvangen
            await asyncio.sleep(0.5)
            
            # Sluit alle actieve verbindingen
            for client in list(self.clients):
                try:
                    await client.close()
                except:
                    pass
        
        # Stop de server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        logger.info('WebSocket server gestopt')
        self.server = None
        self.clients.clear()

class HTTPServer:
    """Eenvoudige HTTP server voor het serveren van statische bestanden met verbeterde foutafhandeling."""
    
    def __init__(self, host='0.0.0.0', port=8000):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False
        self.original_port = port  # Bewaar de originele poort voor foutmeldingen
    
    class RequestHandler(http.server.SimpleHTTPRequestHandler):
        """Aangepaste request handler voor het serveren van bestanden."""
        
        def __init__(self, *args, **kwargs):
            # Stel de root directory in op de map van het script
            directory = os.path.dirname(os.path.abspath(__file__))
            super().__init__(*args, directory=directory, **kwargs)
        
        def end_headers(self):
            """Voeg CORS headers toe aan alle reacties."""
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
        
        def do_OPTIONS(self):
            """Verwerk OPTIONS verzoeken voor CORS preflight."""
            self.send_response(200)
            self.end_headers()
        
        def log_message(self, format, *args):
            """Onderduk standaard logging van de HTTP server."""
            logger.debug(f"HTTP {self.address_string()} - {format % args}")
    
    def find_available_port(self, start_port: int, max_attempts: int = 10) -> Optional[int]:
        """Vind een beschikbare poort vanaf de opgegeven poort."""
        for port in range(start_port, start_port + max_attempts):
            if not is_port_in_use(port, self.host):
                return port
        return None

    def start(self) -> bool:
        """
        Start de HTTP server in een aparte thread.
        
        Returns:
            bool: True als de server is gestart, False als dit niet gelukt is
        """
        if self.running:
            logger.warning('HTTP server is al gestart')
            return True

        try:
            # Controleer of de poort beschikbaar is
            if is_port_in_use(self.port, self.host):
                # Probeer een alternatieve poort te vinden
                new_port = self.find_available_port(self.port + 1)
                if new_port is not None:
                    logger.warning(f'Poort {self.port} is al in gebruik, probeer poort {new_port}...')
                    self.port = new_port
                else:
                    raise OSError(
                        f'Kan geen beschikbare poort vinden in het bereik {self.port}-{self.port + 10}.'
                    )
            
            self.server = socketserver.TCPServer(
                (self.host, self.port),
                self.RequestHandler,
                bind_and_activate=False  # Stel handmatig in om betere foutafhandeling te krijgen
            )
            
            # Stel socket opties in om adreshergebruik mogelijk te maken
            self.server.allow_reuse_address = True
            
            # Bind de socket handmatig voor betere foutafhandeling
            try:
                self.server.server_bind()
                self.server.server_activate()
            except OSError as e:
                if e.errno == 10048:  # WSAEADDRINUSE
                    logger.error(f'Poort {self.port} is al in gebruik. Probeer een andere poort of sluit de vorige instantie af.')
                    return False
                raise

            self.server_thread = threading.Thread(
                target=self.server.serve_forever,
                name=f'HTTP-Server-{self.port}'
            )
            self.server_thread.daemon = True
            self.running = True
            self.server_thread.start()

            logger.info(f'✅ HTTP server gestart op http://{self.host}:{self.port}')
            
            # Controleer of de server echt draait
            if not self.server_thread.is_alive():
                raise RuntimeError('HTTP server thread is niet gestart')
                
            return True

        except OSError as e:
            if e.errno == 10048:  # WSAEADDRINUSE
                logger.error(f'Poort {self.port} is al in gebruik. Probeer een andere poort of sluit de vorige instantie af.')
            else:
                logger.error(f'Fout bij starten HTTP server: {e}')
            self.running = False
            return False
            
        except Exception as e:
            logger.error(f'Onverwachte fout bij starten HTTP server: {e}', exc_info=True)
            self.running = False
            return False

    def stop(self):
        """Stop de HTTP server."""
        if not self.running or not self.server:
            return

        logger.info('Stoppen van HTTP server...')
        self.running = False

        # Stop de server
        self.server.shutdown()
        self.server.server_close()

        # Wacht tot de thread is gestopt
        if self.server_thread:
            self.server_thread.join(timeout=5)

        logger.info('HTTP server gestopt')
        self.server = None
        self.server_thread = None


class JarvisApplication:
    """Hoofdapplicatie die de WebSocket en HTTP servers beheert."""
    
    def __init__(self, ws_host='127.0.0.1', ws_port=18861, http_host='127.0.0.1', http_port=18080, enable_ws=True, enable_http=True):
        """
        Initialiseer de applicatie.
        
        Args:
            ws_host (str): Host voor de WebSocket server (standaard: 127.0.0.1)
            ws_port (int): Poort voor de WebSocket server (standaard: 18861)
            http_host (str): Host voor de HTTP server (standaard: 127.0.0.1)
            http_port (int): Poort voor de HTTP server (standaard: 18080)
            enable_ws (bool): Schakel WebSocket server in (standaard: True)
            enable_http (bool): Schakel HTTP server in (standaard: True)
        """
        
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.http_host = http_host
        self.http_port = http_port
        self.enable_ws = enable_ws
        self.enable_http = enable_http
        
        self.ws_server = WebSocketServer(ws_host, ws_port) if enable_ws else None
        self.http_server = HTTPServer(http_host, http_port) if enable_http else None
        
        self.loop = None
        self.shutdown_event = asyncio.Event()
        self.start_time = None
        
    def print_banner(self):
        """Print een informatieve banner bij het opstarten."""
        # Bepaal de juiste hostnamen voor weergave
        ws_display_host = 'localhost' if self.ws_host in ('0.0.0.0', '127.0.0.1') else self.ws_host
        http_display_host = 'localhost' if self.http_host in ('0.0.0.0', '127.0.0.1') else self.http_host
        
        # Maak de banner met de juiste poorten
        banner = f"""
        ╔══════════════════════════════════════════════╗
        ║           J.A.R.V.I.S. AI Server              ║
        ╠══════════════════════════════════════════════╣"""
        
        if self.enable_http:
            banner += f"\n        ║  • HTTP:   http://{http_display_host}:{self.http_port}"
        if self.enable_ws:
            banner += f"\n        ║  • WebSocket: ws://{ws_display_host}:{self.ws_port}"
            
        banner += """
        ╠══════════════════════════════════════════════╣
        ║  Gebruik Ctrl+C om te stoppen               ║
        ╚══════════════════════════════════════════════╝
        """
        
        try:
            print(banner)
        except UnicodeEncodeError:
            # Fallback voor terminals die geen UTF-8 ondersteunen
            print("\nJ.A.R.V.I.S. AI Server")
            print("======================\n")
            if self.enable_http:
                print(f"HTTP:      http://{http_display_host}:{self.http_port}")
            if self.enable_ws:
                print(f"WebSocket: ws://{ws_display_host}:{self.ws_port}")
            print("\nDruk op Ctrl+C om te stoppen\n")
    
    async def start(self):
        """Start de applicatie."""
        self.start_time = dt.now()
        self.loop = asyncio.get_running_loop()
        
        # Toon de banner
        self.print_banner()
        
        # Registreer signalen voor nette afsluiting (platform-specifiek)
        self._register_signal_handlers()
        
        try:
            # Start de servers onafhankelijk van elkaar
            tasks = []
            
            if self.enable_http:
                tasks.append(asyncio.create_task(self._start_http_server()))
            
            if self.enable_ws:
                tasks.append(asyncio.create_task(self._start_websocket_server()))
            
            if not tasks:
                logger.error('Geen servers ingeschakeld. Sluit af.')
                await self.shutdown()
                return
                
            # Wacht tot alle taken voltooid zijn of tot een fout optreedt
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Controleer of er fouten zijn opgetreden
            for task in done:
                if task.exception():
                    logger.error(f'Fout bij het starten van een server: {task.exception()}')
            
            # Wacht tot de shutdown event is gezet
            await self.shutdown_event.wait()
            
        except asyncio.CancelledError:
            logger.info('Ontvangen annuleringssignaal, sluit af...')
        except Exception as e:
            logger.error(f'Onverwachte fout bij het starten van de applicatie: {e}', exc_info=True)
            await self.shutdown()
    
    def _register_signal_handlers(self):
        """Registreer signaalhandlers voor nette afsluiting."""
        if hasattr(signal, 'SIGINT') and hasattr(signal, 'SIGTERM'):
            try:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    if hasattr(signal, 'SIG_DFL') and hasattr(self.loop, 'add_signal_handler'):
                        try:
                            self.loop.add_signal_handler(
                                sig,
                                lambda s=sig: asyncio.create_task(self.shutdown(s))
                            )
                        except (NotImplementedError, RuntimeError):
                            # Fallback voor Windows of andere platforms zonder signaalondersteuning
                            signal.signal(sig, lambda s, f, sig=sig: asyncio.create_task(self.shutdown(sig)))
            except Exception as e:
                logger.warning(f"Kon signaal handlers niet instellen: {e}")

        # Handmatige afsluiting voor Windows (Ctrl+C)
        if os.name == 'nt' and 'win32api' in sys.modules:
            def win32_signal_handler(sig):
                if sig in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT):
                    # Gebruik een thread-safe manier om de shutdown te starten
                    if hasattr(self, 'loop') and self.loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.shutdown(signal.SIGINT), self.loop)
                    return True
                return False
            
            # Sla de handler op als attribuut om te voorkomen dat deze wordt opgeruimd
            self.win32_handler = win32api.SetConsoleCtrlHandler(win32_signal_handler, True)
    
    async def _start_http_server(self):
        """Start de HTTP server."""
        if not self.enable_http or not self.http_server:
            return False
            
        try:
            started = self.http_server.start()
            if not started:
                logger.warning('HTTP server kon niet worden gestart')
                return False
                
            logger.info(f'✅ HTTP server is actief op http://{self.http_host}:{self.http_port}')
            return True
            
        except Exception as e:
            logger.error(f'Fout bij starten HTTP server: {e}', exc_info=True)
            return False
    
    async def _start_websocket_server(self):
        """Start de WebSocket server."""
        if not self.enable_ws or not self.ws_server:
            return False
            
        try:
            started = await self.ws_server.start()
            if not started:
                logger.warning('WebSocket server kon niet worden gestart')
                return False
                
            logger.info(f'✅ WebSocket server is actief op ws://{self.ws_host}:{self.ws_port}')
            return True
            
        except Exception as e:
            logger.error(f'Fout bij starten WebSocket server: {e}', exc_info=True)
            return False
    
    async def shutdown(self, signal=None):
        """Sluit de applicatie netjes af."""
        if hasattr(self, 'shutdown_event') and self.shutdown_event.is_set():
            return
            
        logger.info(f"\nOntvangen signaal {signal}, sluit af...")
        
        # Markeer dat we aan het afsluiten zijn
        if not hasattr(self, 'shutdown_event'):
            self.shutdown_event = asyncio.Event()
        
        # Zet de shutdown event om de hoofdloop te laten stoppen
        self.shutdown_event.set()
        
        # Geef wat tijd voor het afsluiten van alle taken
        await asyncio.sleep(0.5)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Start de J.A.R.V.I.S. AI server')
    parser.add_argument('--http-port', type=int, default=18080, 
                        help='HTTP server poort (standaard: 18080)')
    parser.add_argument('--ws-port', type=int, default=18861, 
                        help='WebSocket server poort (standaard: 18861)')
    parser.add_argument('--host', type=str, default='127.0.0.1', 
                        help='Bind adres voor beide servers (standaard: 127.0.0.1)')
    parser.add_argument('--no-http', action='store_true', 
                        help='Start geen HTTP server')
    parser.add_argument('--no-ws', action='store_true', 
                        help='Start geen WebSocket server')
    parser.add_argument('--debug', action='store_true',
                        help='Schakel debug logging in')
    return parser.parse_args()

def setup_logging(debug=False):
    """Configureer logging met kleuren en bestandsoutput."""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Maak een formatter met kleuren
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'WARNING': '\033[93m',
            'INFO': '\033[92m',
            'DEBUG': '\033[96m',
            'CRITICAL': '\033[91m',
            'ERROR': '\033[91m',
            'RED': '\033[91m',
            'GREEN': '\033[92m',
            'YELLOW': '\033[93m',
            'BLUE': '\033[94m',
            'RESET': '\033[0m'
        }
        
        def format(self, record):
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                record.msg = f"{self.COLORS.get(levelname, '')}{record.msg}{self.COLORS['RESET']}"
            return super().format(record)
    
    # Configureer de root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Verwijder bestaande handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Voeg console handler toe
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Voeg bestand handler toe
    file_handler = logging.FileHandler('jarvis_server.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

def is_port_in_use(port, host='0.0.0.0'):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def run_application():
    """Start de applicatie met command line argumenten."""
    args = parse_arguments()
    setup_logging(debug=args.debug)
    
    logger = logging.getLogger('JarvisMain')
    
    try:
        # Controleer of er ten minste één server is ingeschakeld
        if args.no_http and args.no_ws:
            logger.error("Fout: Beide servers zijn uitgeschakeld met --no-http en --no-ws")
            return 1
        
        # Functie om een beschikbare poort te vinden
        def find_available_port(start_port, host, max_attempts=10):
            port = start_port
            attempts = 0
            while attempts < max_attempts:
                if not is_port_in_use(port, host):
                    return port
                port += 1
                attempts += 1
            return None
        
        # Controleer en pas poorten aan indien nodig
        if not args.no_http and is_port_in_use(args.http_port, args.host):
            logger.warning(f"Waarschuwing: Poort {args.http_port} is in gebruik, zoek naar een beschikbare poort...")
            new_http_port = find_available_port(args.http_port + 1, args.host)
            if new_http_port is None:
                logger.error(f"Geen beschikbare HTTP poort gevonden na meerdere pogingen")
                return 1
            logger.info(f"Gebruik nu HTTP poort {new_http_port}")
            args.http_port = new_http_port
            
        if not args.no_ws and is_port_in_use(args.ws_port, args.host):
            logger.warning(f"Waarschuwing: Poort {args.ws_port} is in gebruik, zoek naar een beschikbare poort...")
            new_ws_port = find_available_port(args.ws_port + 1, args.host)
            if new_ws_port is None:
                logger.error(f"Geen beschikbare WebSocket poort gevonden na meerdere pogingen")
                return 1
            logger.info(f"Gebruik nu WebSocket poort {new_ws_port}")
            args.ws_port = new_ws_port
        
        # Maak een nieuwe event loop voor deze thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Log de gebruikte instellingen
        logger.info(f"\nStartende J.A.R.V.I.S. server met de volgende instellingen:")
        logger.info(f"  - Host: {args.host}")
        logger.info(f"  - HTTP poort: {args.http_port} {'(uitgeschakeld)' if args.no_http else '(ingeschakeld)'}")
        logger.info(f"  - WebSocket poort: {args.ws_port} {'(uitgeschakeld)' if args.no_ws else '(ingeschakeld)'}")
        logger.info("Druk op Ctrl+C om te stoppen\n")
        
        # Maak en start de applicatie
        try:
            app = JarvisApplication(
                ws_host=args.host,
                ws_port=args.ws_port,
                http_host=args.host,
                http_port=args.http_port,
                enable_ws=not args.no_ws,
                enable_http=not args.no_http
            )
        except Exception as e:
            logger.error(f"Kon de applicatie niet starten: {e}")
            return 1
        
        # Run de applicatie in de event loop
        try:
            loop.run_until_complete(app.start())
        except KeyboardInterrupt:
            logger.info('\nOntvangen Ctrl+C, sluit netjes af...')
            loop.run_until_complete(app.shutdown())
        except Exception as e:
            logger.error(f'Onverwachte fout: {e}', exc_info=True)
            return 1
        finally:
            # Sluit de event loop netjes af
            pending = [t for t in asyncio.all_tasks(loop=loop) 
                     if not t.done() and not t.cancelled()]
            
            if pending:
                logger.info(f'Annuleer {len(pending)} lopende taken...')
                for task in pending:
                    task.cancel()
                
                # Verzamel eventuele fouten van geannuleerde taken
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
            # Sluit alle asyncio transporten
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            
            # Sluit de event loop
            if not loop.is_closed():
                loop.close()
            
            logger.info('Applicatie afgesloten')
        
        return 0
        
    except Exception as e:
        logger.critical(f'Kritieke fout: {e}', exc_info=True)
        return 1

if __name__ == '__main__':
    try:
        # Voeg de huidige map toe aan het pad voor imports
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Start de applicatie en sla de exit code op
        exit_code = run_application()
        
        # Geef een korte samenvatting van de beschikbare opties bij een fout
        if exit_code != 0:
            print("\nGebruik: python test_jarvis_integration.py [OPTIES]")
            print("Opties:")
            print("  --http-port PORT   Poort voor de HTTP server (standaard: 18080)")
            print("  --ws-port PORT     Poort voor de WebSocket server (standaard: 18861)")
            print("  --host HOST        Bind adres (standaard: 127.0.0.1)")
            print("  --no-http         Start geen HTTP server")
            print("  --no-ws           Start geen WebSocket server")
            print("  --debug           Schakel gedetailleerde logging in")
            print("\nVoorbeeld: python test_jarvis_integration.py --http-port 18081 --ws-port 18862")
        
        # Sluit af met de juiste exit code
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\nApplicatie onderbroken door gebruiker")
        sys.exit(0)
    except Exception as e:
        logger.critical(f'Onverwachte fout: {e}', exc_info=True)
        sys.exit(1)
