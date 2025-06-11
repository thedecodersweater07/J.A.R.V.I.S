"""
Integration tests for the JARVIS model.

This script provides a simple way to test the core functionality of the JARVIS model.
"""

import unittest
import sys
import os
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the models directly from the package
try:
    from models.jarvis import JarvisModel, JarvisLanguageModel, create_jarvis_model
except ImportError as e:
    print(f"Error importing JARVIS modules: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Models directory exists: {os.path.exists(os.path.join(project_root, 'models'))}")
    raise

# Simple HTTP server to serve the web interface
class JARVISRequestHandler(http.server.SimpleHTTPRequestHandler):
    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        try:
            # Handle API endpoints
            if self.path == '/api/health':
                self._set_headers()
                self.wfile.write(b'{"status": "ok", "service": "JARVIS Test Server"}')
                return
                
            elif self.path == '/api/stats':
                self._set_headers()
                self.wfile.write(b'{"active_connections": 1, "status": "running"}')
                return
                
            # Handle chat history
            elif self.path == '/api/chat/history':
                self._set_headers()
                self.wfile.write(b'{"messages": []}')
                return
                
            # Handle shutdown request
            elif self.path == '/shutdown':
                self._set_headers(content_type='text/plain')
                self.wfile.write(b'Server is shutting down...')
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
                
            # Default to serving files
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
            
        except (ConnectionResetError, BrokenPipeError):
            # Ignore connection errors during shutdown
            pass
    
    def do_POST(self):
        try:
            if self.path == '/api/chat/send':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Parse the incoming message
                import json
                data = json.loads(post_data)
                message = data.get('message', '').strip()
                
                if not message:
                    self._set_headers(400)
                    self.wfile.write(b'{"error": "Message is required"}')
                    return
                
                # Process the message with JARVIS
                response = self.process_message(message)
                
                # Send the response
                self._set_headers()
                self.wfile.write(json.dumps({
                    'response': response,
                    'timestamp': time.time()
                }).encode('utf-8'))
                return
                
            self._set_headers(404)
            self.wfile.write(b'{"error": "Endpoint not found"}')
            
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(b'{"error": "Invalid JSON"}')
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(f'{{"error": "Internal server error: {str(e)}"}}'.encode('utf-8'))
    
    def process_message(self, message: str) -> str:
        """Process a message using JARVIS and return the response."""
        # Simple response for now - in a real app, this would use the LLM
        responses = [
            f"I received your message: {message}",
            f"You said: {message}",
            f"Interesting point about '{message}'. Can you tell me more?",
            f"I'm processing your message: {message}",
            f"Thanks for sharing: {message}"
        ]
        import random
        return random.choice(responses)


class SimpleServer:
    def __init__(self, port=8000):
        self.port = port
        self.httpd = None
        self.thread = None
        self.running = False
        self.handler = JARVISRequestHandler
    
    def start(self):
        """Start the HTTP server in a separate thread."""
        try:
            # Change to the project root directory to serve files
            os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            print(f"Serving files from: {os.getcwd()}")
            
            # Create server instance with allow_reuse_address=True
            self.httpd = socketserver.TCPServer(('', self.port), self.handler, bind_and_activate=False)
            self.httpd.allow_reuse_address = True
            self.httpd.server_bind()
            self.httpd.server_activate()
            
            # Start server in a separate thread
            self.running = True
            self.thread = threading.Thread(target=self._run_server)
            self.thread.daemon = True
            self.thread.start()
            
            # Server info
            print(f"\nServing HTTP on http://localhost:{self.port}")
            print("Press Ctrl+C to stop the server\n")
            
            # Open browser automatically
            try:
                webbrowser.open(f"http://localhost:{self.port}/index.html")
            except Exception as e:
                print(f"Could not open browser: {e}")
                
        except Exception as e:
            print(f"Failed to start server: {e}")
            self.running = False
            raise
    
    def _run_server(self):
        """Run the HTTP server."""
        try:
            print("Server thread started")
            self.httpd.serve_forever()
            print("Server finished serving")
        except Exception as e:
            if self.running:  # Only log if we didn't stop on purpose
                print(f"Server error: {e}")
        finally:
            self.running = False
    
    def stop(self):
        """Stop the HTTP server."""
        if not self.running or not self.httpd:
            return
            
        print("Shutting down server...")
        self.running = False
        
        try:
            # Send a shutdown request to the server
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                try:
                    s.connect(('localhost', self.port))
                    s.sendall(b'GET /shutdown HTTP/1.1\r\n\r\n')
                    # Wait for the response
                    s.recv(1024)
                except (socket.timeout, ConnectionRefusedError):
                    # Server might have already shut down
                    pass
            
            # Wait a bit for the server to process the shutdown
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2.0)
                
        except Exception as e:
            print(f"Error during server shutdown: {e}")
        finally:
            # Ensure cleanup
            try:
                if hasattr(self, 'httpd') and self.httpd:
                    self.httpd.server_close()
            except Exception as e:
                print(f"Error during server cleanup: {e}")
            finally:
                self.httpd = None
                print("Server shutdown complete")

class TestJarvisModel(unittest.TestCase):
    """Test cases for the JARVIS model."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        print("\nSetting up test environment...")
        try:
            # Initialize the language model
            cls.language_model = create_jarvis_model("language", "jarvis-base")
            print("✅ JARVIS language model initialized successfully")
            
            # Skip tests if LLM service is not available
            if not hasattr(cls.language_model, 'llm_service') or cls.language_model.llm_service is None:
                print("⚠️  LLM service not available, some tests will be skipped")
                cls.skip_llm_tests = True
            else:
                cls.skip_llm_tests = False
                
        except Exception as e:
            print(f"❌ Failed to initialize JARVIS language model: {e}")
            cls.skip_llm_tests = True

    def test_language_generation(self):
        """Test basic text generation."""
        if self.skip_llm_tests:
            self.skipTest("Skipping language generation test - LLM service not available")
            
        prompt = "Hello, how are you today?"
        try:
            response = self.language_model.generate(prompt, max_length=50)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            print(f"✅ Language generation test passed. Response: {response[:100]}...")
        except Exception as e:
            print(f"❌ Language generation test failed: {e}")
            if not self.skip_llm_tests:  # Only raise if we expected LLM to be available
                raise

    def test_text_classification(self):
        """Test text classification."""
        if self.skip_llm_tests:
            self.skipTest("Skipping text classification test - LLM service not available")
            
        print("\nTesting text classification...")
        try:
            # This method might not be implemented yet, so we'll skip it for now
            if not hasattr(self.language_model, 'classify'):
                self.skipTest("Text classification method not implemented")
                
            categories = self.language_model.classify("I'm feeling great today!")
            self.assertIsInstance(categories, dict)
            self.assertGreater(len(categories), 0)
            print(f"✅ Text classification test passed. Categories: {categories}")
        except Exception as e:
            print(f"❌ Text classification test failed: {e}")
            if not self.skip_llm_tests:
                raise

    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        if self.skip_llm_tests:
            self.skipTest("Skipping sentiment analysis test - LLM service not available")
            
        print("\nTesting sentiment analysis...")
        try:
            # This method might not be implemented yet, so we'll skip it for now
            if not hasattr(self.language_model, 'analyze_sentiment'):
                self.skipTest("Sentiment analysis method not implemented")
                
            sentiment = self.language_model.analyze_sentiment("I love this product!")
            self.assertIsInstance(sentiment, dict)
            self.assertIn('sentiment', sentiment)
            print(f"✅ Sentiment analysis test passed. Sentiment: {sentiment}")
        except Exception as e:
            print(f"❌ Sentiment analysis test failed: {e}")
            if not self.skip_llm_tests:
                raise

def main():
    print("\n" + "="*50)
    print("Starting JARVIS Test Server")
    print("="*50 + "\n")
    
    # Start the HTTP server
    server = SimpleServer(port=8000)
    server.start()
    
    # Run tests without blocking
    run_tests()
    
    print("\nTests completed. Server is still running...")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Keep the server running until interrupted
        while server.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Cleanup
        server.stop()
        print("\n" + "="*50)
        print("Server stopped")
        print("="*50)


def run_tests():
    """Run the test suite."""
    try:
        unittest.main(exit=False)
    except Exception as e:
        print(f"Error running tests: {e}")


if __name__ == "__main__":
    import time
    main()
