import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import logging
from typing import Optional, Callable, Dict, Any
from threading import Thread
import time

from ui.components.chat_history import ChatHistory
from ui.components.chat_input import ChatInput
from ui.components.status_bar import StatusBar
# from ui.components.message_bubble import MessageBubble  # Uitgeschakeld, niet gevonden
from ui.themes.theme_manager import ThemeManager

class Screen:
    """Moderne JARVIS UI Screen met geavanceerde features"""
    
    def __init__(self, model=None, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        # UI Properties
        self.width = self.config.get('width', 1000)
        self.height = self.config.get('height', 700)
        self.title = self.config.get('title', 'JARVIS AI Assistant')
        # Tkinter components
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry(f"{self.width}x{self.height}")
        self.root.minsize(800, 600)
        self.main_frame = None
        self.sidebar = None
        # Custom components
        self.chat_history = None
        self.chat_input = None
        self.status_bar = None
        self.theme_manager = ThemeManager()
        # Callbacks
        self.on_send: Optional[Callable[[str], None]] = None
        self.on_voice_command = None  # type: ignore
        # State
        self.is_running = False
        self.typing_indicator = False
        # Immediately initialize the UI
        self.initialize()

    def initialize(self) -> bool:
        """Initialiseer de UI"""
        try:
            # self.root = tk.Tk()  # Already created in __init__
            # self.root.title(self.title)
            # self.root.geometry(f"{self.width}x{self.height}")
            # self.root.minsize(800, 600)
            # Load en apply theme
            self.theme_manager.load_themes()
            current_theme = self.config.get('theme', 'stark')
            self.theme_manager.apply_theme(current_theme)
            
            self._setup_styles()
            self._build_ui()
            self._setup_bindings()
            
            self.logger.info("JARVIS UI geïnitialiseerd")
            return True
            
        except Exception as e:
            self.logger.error(f"Fout bij UI initialisatie: {e}")
            return False

    def _setup_styles(self):
        """Setup moderne styling"""
        style = ttk.Style()
        theme = self.theme_manager.get_current_theme()
        
        # Configure styles
        style.configure('Main.TFrame', 
                       background=theme.get('background', '#1a1a1a'))
        
        style.configure('Sidebar.TFrame',
                       background=theme.get('sidebar', '#2d2d2d'))
        
        style.configure('Modern.TButton',
                       background=theme.get('primary', '#00d4ff'),
                       foreground=theme.get('text', '#ffffff'),
                       borderwidth=0,
                       focuscolor='none')
        
        style.map('Modern.TButton',
                 background=[('active', theme.get('primary_hover', '#0099cc'))])

    def _build_ui(self):
        """Bouw de complete UI"""
        # Main container
        self.main_frame = ttk.Frame(self.root, style='Main.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create layout
        self._create_sidebar()
        self._create_chat_area()
        self._create_input_area()
        self._create_status_bar()

    def _create_sidebar(self):
        """Maak sidebar met controls"""
        self.sidebar = ttk.Frame(self.main_frame, style='Sidebar.TFrame', width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.sidebar.pack_propagate(False)
        
        # Sidebar content
        ttk.Label(self.sidebar, text="JARVIS", 
                 font=('Arial', 16, 'bold'),
                 background=self.theme_manager.get_color('sidebar')).pack(pady=10)
        
        # Theme selector
        theme_frame = ttk.Frame(self.sidebar)
        theme_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(theme_frame, text="Theme:").pack(anchor=tk.W)
        self.theme_var = tk.StringVar(value=self.config.get('theme', 'stark'))
        theme_combo = ttk.Combobox(theme_frame, textvariable=self.theme_var,
                                  values=['stark', 'minimal', 'dark', 'light'],
                                  state='readonly')
        theme_combo.pack(fill=tk.X, pady=2)
        theme_combo.bind('<<ComboboxSelected>>', self._on_theme_change)
        
        # Voice toggle
        self.voice_var = tk.BooleanVar(value=True)
        voice_check = ttk.Checkbutton(self.sidebar, text="Voice Input",
                                     variable=self.voice_var)
        voice_check.pack(anchor=tk.W, padx=10, pady=5)
        
        # Clear chat button
        clear_btn = ttk.Button(self.sidebar, text="Clear Chat",
                              command=self._clear_chat,
                              style='Modern.TButton')
        clear_btn.pack(fill=tk.X, padx=10, pady=5)

    def _create_chat_area(self):
        """Maak chat gebied"""
        chat_frame = ttk.Frame(self.main_frame)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Chat history
        self.chat_history = ChatHistory(chat_frame, self.theme_manager)
        self.chat_history.pack(fill=tk.BOTH, expand=True)

    def _create_input_area(self):
        """Maak input gebied"""
        input_frame = ttk.Frame(self.main_frame)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.chat_input = ChatInput(input_frame, self.theme_manager,
                                   on_send=self._handle_send,
                                   on_voice=self._handle_voice)
        self.chat_input.pack(fill=tk.X)

    def _create_status_bar(self):
        """Maak status bar"""
        self.status_bar = StatusBar(self.main_frame, self.theme_manager)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar.set_status("JARVIS UI Ready")

    def _setup_bindings(self):
        """Setup keyboard shortcuts"""
        if self.root is not None:
            self.root.bind('<Control-q>', lambda e: self.shutdown())
            self.root.bind('<Control-l>', lambda e: self._clear_chat())
            self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

    def _handle_send(self, message: str):
        """Handle verzonden bericht"""
        if not message.strip():
            return
            
        # Add user message
        if self.chat_history and hasattr(self.chat_history, 'add_message'):
            self.chat_history.add_message("You", message, "user")
        
        # Show typing indicator
        self._show_typing_indicator()
        
        # Process with model
        if self.model:
            Thread(target=self._process_ai_response, args=(message,), daemon=True).start()
        elif self.on_send:
            self.on_send(message)
        else:
            self._hide_typing_indicator()
            if self.chat_history and hasattr(self.chat_history, 'add_message'):
                self.chat_history.add_message("JARVIS", "No AI model connected", "assistant")

    def _process_ai_response(self, message: str):
        """Process AI response in thread"""
        try:
            response = self.model.generate(message) if self.model else None
            
            # Parse response
            if isinstance(response, dict):
                text = response.get('text') or response.get('response') or str(response)
            else:
                text = str(response)
            
            # Update UI in main thread
            if self.root is not None:
                self.root.after(0, self._display_ai_response, text)
            
        except Exception as e:
            error_msg = f"AI Error: {str(e)}"
            if self.root is not None:
                self.root.after(0, self._display_ai_response, error_msg)

    def _display_ai_response(self, response: str):
        """Display AI response"""
        self._hide_typing_indicator()
        if self.chat_history and hasattr(self.chat_history, 'add_message'):
            self.chat_history.add_message("JARVIS", response, "assistant")

    def _show_typing_indicator(self):
        """Toon typing indicator"""
        self.typing_indicator = True
        if self.chat_history and hasattr(self.chat_history, 'show_typing_indicator'):
            self.chat_history.show_typing_indicator()

    def _hide_typing_indicator(self):
        """Verberg typing indicator"""
        self.typing_indicator = False
        if self.chat_history and hasattr(self.chat_history, 'hide_typing_indicator'):
            self.chat_history.hide_typing_indicator()

    def _handle_voice(self):
        """Handle voice input"""
        if self.voice_var.get() and self.status_bar is not None:
            self.status_bar.set_status("Listening...")
            # Voice input logic here
            if self.root is not None:
                self.root.after(2000, lambda: self.status_bar.set_status("Ready") if self.status_bar is not None else None)

    def _on_theme_change(self, event=None):
        """Handle theme change"""
        new_theme = self.theme_var.get()
        if self.theme_manager.apply_theme(new_theme):
            self._setup_styles()
            self._refresh_components()

    def _refresh_components(self):
        """Refresh all components with new theme"""
        if self.chat_history and hasattr(self.chat_history, 'apply_theme'):
            self.chat_history.apply_theme()
        if self.chat_input and hasattr(self.chat_input, 'apply_theme'):
            self.chat_input.apply_theme()
        if self.status_bar and hasattr(self.status_bar, 'apply_theme'):
            self.status_bar.apply_theme()

    def _clear_chat(self):
        """Clear chat history"""
        if self.chat_history and messagebox.askyesno("Clear Chat", "Are you sure you want to clear the chat history?"):
            if hasattr(self.chat_history, 'clear'):
                self.chat_history.clear()

    def add_message(self, sender: str, message: str, msg_type: str = "user"):
        """Voeg bericht toe aan chat"""
        if self.chat_history and hasattr(self.chat_history, 'add_message'):
            self.chat_history.add_message(sender, message, msg_type)

    def set_status(self, status: str):
        """Set status text"""
        if self.status_bar and hasattr(self.status_bar, 'set_status'):
            self.status_bar.set_status(status)

    def run(self):
        """Start de UI"""
        if not self.is_running:
            self.is_running = True
            self.logger.info("Starting JARVIS UI...")
            if self.root:
                self.root.mainloop()

    def shutdown(self):
        """Sluit UI af"""
        if self.is_running:
            self.is_running = False
            self.logger.info("Shutting down JARVIS UI...")
            if self.root:
                self.root.quit()
                self.root.destroy()