from typing import List, Any

class TerminalRenderer:
    """
    A class responsible for rendering output to the terminal.
    """
    
    def __init__(self):
        self.clear_sequence = "\033[H\033[J"  # ANSI escape sequence to clear screen
        
    def clear_screen(self):
        """Clears the terminal screen."""
        print(self.clear_sequence, end='')
        
    def render_text(self, text: str):
        """
        Renders plain text to the terminal.
        
        Args:
            text (str): The text to render
        """
        print(text)
        
    def render_list(self, items: List[Any], numbered: bool = False):
        """
        Renders a list of items to the terminal.
        
        Args:
            items (List[Any]): List of items to render
            numbered (bool): If True, adds numbers to the list items
        """
        for idx, item in enumerate(items, 1):
            if numbered:
                print(f"{idx}. {str(item)}")
            else:
                print(f"• {str(item)}")
                
    def render_error(self, message: str):
        """
        Renders an error message in red.
        
        Args:
            message (str): The error message to display
        """
        print(f"\033[91m{message}\033[0m")
        
    def render_success(self, message: str):
        """
        Renders a success message in green.
        
        Args:
            message (str): The success message to display
        """
        print(f"\033[92m{message}\033[0m")