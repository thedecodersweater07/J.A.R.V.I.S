import tkinter as tk
from ...llm.manager import LLMManager
from ...data.db.database import DatabaseManager

class ChatWindow(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.llm_manager = LLMManager()
        self.db = DatabaseManager()
        self.setup_ui()
        
    def setup_ui(self):
        self.chat_area = tk.Text(self, height=20)
        self.input_field = tk.Entry(self)
        self.send_button = tk.Button(self, text="Send", command=self.send_message)
        
        self.chat_area.pack(pady=5)
        self.input_field.pack(pady=5)
        self.send_button.pack()
        
    async def send_message(self):
        user_input = self.input_field.get()
        response = await self.llm_manager.process_input(user_input, "current_user")
        self.display_message(user_input, response)
