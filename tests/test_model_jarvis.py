from flask import Flask, render_template, request, jsonify, send_from_directory
import time
import sys
import os
from typing import Dict, Any, List

app = Flask(__name__)

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import your actual Jarvis model
try:
    # Try different import methods
    if os.path.exists(os.path.join(current_dir, 'jarvis.py')):
        print("🔍 Found jarvis.py in current directory")
        import jarvis
        
        # Check if JarvisModel class exists
        if hasattr(jarvis, 'JarvisModel'):
            JarvisModel = jarvis.JarvisModel
            print("✅ Successfully imported JarvisModel from jarvis.py")
            USE_REAL_MODEL = True
        else:
            # Look for other possible class names
            possible_classes = [name for name in dir(jarvis) 
                              if not name.startswith('_') and callable(getattr(jarvis, name))]
            if possible_classes:
                print(f"🔍 Found these classes in jarvis.py: {possible_classes}")
                # Use the first class found
                JarvisModel = getattr(jarvis, possible_classes[0])
                print(f"✅ Using {possible_classes[0]} as the model class")
                USE_REAL_MODEL = True
            else:
                raise ImportError("No suitable class found in jarvis.py")
    else:
        raise ImportError("jarvis.py not found in current directory")
        
except ImportError as e:
    print(f"❌ Could not import jarvis.py: {e}")
    print("📝 Please make sure jarvis.py is in the same directory as this server file")
    print(f"📁 Current directory: {current_dir}")
    print(f"📁 Files in directory: {[f for f in os.listdir(current_dir) if f.endswith('.py')]}")
    print("🔄 Using mock model for testing")
    USE_REAL_MODEL = False
    
    # Fallback Mock AI Model (only if real model not available)
    class JarvisModel:
        def __init__(self):
            self.model_state = "ready"
            self.conversation_history = []
            self.test_results = []
            self.is_processing = False
        
        def process_input(self, user_input: str) -> str:
            """Process user input and return AI response"""
            self.is_processing = True
            
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": time.time(),
                "user": user_input,
                "response": None
            })
            
            # Simulate processing time
            time.sleep(0.5)
            
            # Generate mock response based on input
            response = self._generate_response(user_input)
            
            # Update conversation history
            self.conversation_history[-1]["response"] = response
            
            self.is_processing = False
            return response
        
        def _generate_response(self, input_text: str) -> str:
            """Generate mock AI response"""
            input_lower = input_text.lower()
            
            # Simple response patterns
            if "hello" in input_lower or "hi" in input_lower:
                return "Hello! I'm Jarvis, your AI assistant. How can I help you today?"
            
            elif "how are you" in input_lower:
                return "I'm functioning optimally! All systems are running smoothly."
            
            elif "weather" in input_lower:
                return "I don't have access to real-time weather data, but I can help you with other tasks!"
            
            elif "test" in input_lower:
                return "Test mode activated. All systems are responding normally. What would you like to test?"
            
            elif "status" in input_lower:
                return f"System Status: {self.model_state.title()}\nConversations: {len(self.conversation_history)}\nProcessing: {'Yes' if self.is_processing else 'No'}"
            
            elif "help" in input_lower:
                return "I can help you with: conversations, system status, testing, and general questions. Just type your message!"
            
            elif len(input_text.strip()) == 0:
                return "I notice you sent an empty message. Please type something for me to respond to!"
            
            else:
                # General response for other inputs
                responses = [
                    f"I understand you're saying: '{input_text}'. That's an interesting point!",
                    f"Thanks for sharing that. Regarding '{input_text}', I'd be happy to help you explore this further.",
                    f"I've processed your input about '{input_text}'. What specific aspect would you like to discuss?",
                    f"Your message '{input_text}' has been received and analyzed. How can I assist you with this?"
                ]
                import random
                return random.choice(responses)
        
        def get_conversation_history(self) -> List[Dict]:
            """Get conversation history"""
            return self.conversation_history
        
        def clear_history(self):
            """Clear conversation history"""
            self.conversation_history.clear()
        
        def get_stats(self) -> Dict[str, Any]:
            """Get model statistics"""
            return {
                "state": self.model_state,
                "total_conversations": len(self.conversation_history),
                "is_processing": self.is_processing,
                "uptime": time.time() - start_time
            }

# Initialize model
if USE_REAL_MODEL:
    print("🚀 Initializing real Jarvis model...")
    try:
        model = JarvisModel()
        print("✅ Jarvis model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing real model: {e}")
        print("🔄 Falling back to mock model...")
        USE_REAL_MODEL = False
        model = JarvisModel()
else:
    print("🔄 Using mock model...")
    model = JarvisModel()

start_time = time.time()

# Helper function to safely call model methods
def safe_model_call(method_name, *args, **kwargs):
    """Safely call model methods with fallback"""
    try:
        if hasattr(model, method_name):
            return getattr(model, method_name)(*args, **kwargs)
        else:
            print(f"⚠️ Method {method_name} not found in model")
            return f"Method {method_name} not available"
    except Exception as e:
        print(f"❌ Error calling {method_name}: {e}")
        return f"Error: {str(e)}"

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message.strip():
            return jsonify({'error': 'Empty message'}), 400
        
        print(f"📝 User: {user_message}")
        
        # Try to use the real model's method or fallback to mock
        if USE_REAL_MODEL and hasattr(model, 'process_input'):
            response = safe_model_call('process_input', user_message)
        elif USE_REAL_MODEL and hasattr(model, 'generate_response'):
            response = safe_model_call('generate_response', user_message)
        elif USE_REAL_MODEL and hasattr(model, 'chat'):
            response = safe_model_call('chat', user_message)
        elif USE_REAL_MODEL and hasattr(model, 'respond'):
            response = safe_model_call('respond', user_message)
        else:
            # Use mock response
            response = model.process_input(user_message)
        
        print(f"🤖 Jarvis: {response}")
        return jsonify({'response': str(response)})
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return jsonify({'error': f'Model error: {str(e)}'}), 500

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    try:
        if hasattr(model, 'clear_history'):
            safe_model_call('clear_history')
        print("🧹 Chat history cleared")
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get model statistics"""
    try:
        if hasattr(model, 'get_stats'):
            stats = safe_model_call('get_stats')
        else:
            # Basic stats fallback
            stats = {
                'state': 'running',
                'total_conversations': 0,
                'is_processing': False,
                'uptime': time.time() - start_time
            }
        
        stats['model_type'] = 'Real Jarvis Model' if USE_REAL_MODEL else 'Mock Model'
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': USE_REAL_MODEL,
        'uptime': time.time() - start_time,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("🌐 Starting Jarvis AI Model Test Server...")
    print(f"📡 Server will be available at: http://localhost:5000")
    print(f"📄 Make sure index.html is in the same directory!")
    
    if USE_REAL_MODEL:
        print(f"🧠 Using real Jarvis model: {type(model).__name__}")
        # Try to show available methods
        methods = [method for method in dir(model) if not method.startswith('_')]
        print(f"🔧 Available methods: {methods[:5]}...")  # Show first 5 methods
    else:
        print("🎭 Using mock model for testing")
    
    print("🚀 Starting server...")
    app.run(host='0.0.0.0', port=5000, debug=True)