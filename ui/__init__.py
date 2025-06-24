"""
JARVIS UI Module - Moderne interface componenten voor AI assistent
Biedt complete UI-ervaring met theming, voice input, chat interface
"""
from .screen import Screen
from .themes.theme_manager import ThemeManager
from .input.voice_input import VoiceInput
from .input.text_input import TextInput
from .components.chat_history import ChatHistory
from .components.chat_input import ChatInput
from .components.status_bar import StatusBar

__version__ = "2.0.0"
__author__ = "JARVIS UI Team"

__all__ = [
    # Core components
    'Screen',
    'ThemeManager',
    
    # Input components  
    'VoiceInput',
    'TextInput',
    
    # UI Components
    'ChatHistory',
    'ChatInput', 
    'StatusBar',
    
    # Display systems (uitgeschakeld)
    # 'HologramProjector',
    # 'DisplayManager',
    # 'RenderingEngine'
]

# Default configuration
DEFAULT_CONFIG = {
    "window": {
        "width": 1000,
        "height": 700,
        "title": "JARVIS AI Assistant",
        "resizable": True,
        "theme": "stark"
    },
    "chat": {
        "max_messages": 1000,
        "auto_scroll": True,
        "typing_animation": True
    },
    "voice": {
        "enabled": True,
        "wake_word": "jarvis",
        "language": "nl-NL"
    }
}