import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from .renderer_base import RendererBase

logger = logging.getLogger(__name__)

class InputEvent:
    """Simple class to represent input events"""
    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)

# Tkinter rendering is nu standaard via widgets, geen losse renderer meer nodig.
# Deze module kan als placeholder blijven voor toekomstige uitbreidingen.

class TextRenderer:
    def __init__(self):
        pass
    def init(self):
        return True
    def cleanup(self):
        pass