"""
Utility functions and classes for the language learning module.

This module provides various utility classes and functions that support
the language learning functionality, including progress tracking,
data management, and helper functions.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path

__all__ = ['ProgressTracker']

# Import the progress tracker
try:
    from .progress_tracker import ProgressTracker
except ImportError:
    # Create a dummy class if the progress tracker is not available
    class ProgressTracker:
        """Dummy progress tracker for when the real one is not available."""
        def update_vocabulary(self, *args: Any, **kwargs: Any) -> None:
            """Update vocabulary progress (dummy implementation)."""
            pass
            
        def update_grammar(self, *args: Any, **kwargs: Any) -> None:
            """Update grammar progress (dummy implementation)."""
            pass
            
        def update_pronunciation(self, *args: Any, **kwargs: Any) -> None:
            """Update pronunciation progress (dummy implementation)."""
            pass
            
        def get_overall_progress(self) -> Dict[str, float]:
            """Get overall progress (dummy implementation)."""
            return {"vocabulary": 0.0, "grammar": 0.0, "pronunciation": 0.0}
