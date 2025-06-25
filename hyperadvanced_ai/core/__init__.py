"""
Core package for the hyperadvanced_ai framework.

This package contains the core abstractions and implementations for the
hyperadvanced_ai framework, including the main abstraction layer and base classes.
"""

import sys
import importlib
from typing import Any, Type, TypeVar, Generic, TYPE_CHECKING, Dict, Optional

# Import logging functions first (no dependencies)
from hyperadvanced_ai.core.logging import get_logger, configure_logging

# Import ModuleConfig directly
from hyperadvanced_ai.core.base_module import ModuleConfig, BaseAIModule

# Import core types
from hyperadvanced_ai.core.types import (
    AIModule,
    ModuleMetadata,
    ModuleInitializationError,
    T
)

# Import HyperAIAbstraction
from hyperadvanced_ai.core.abstraction import HyperAIAbstraction

# Define __all__ for public API
__all__ = [
    'AIModule',
    'BaseAIModule',
    'HyperAIAbstraction',
    'ModuleConfig',
    'ModuleInitializationError',
    'ModuleMetadata',
    'get_logger',
    'configure_logging',
    'T'
]

# For backward compatibility and runtime type checking
if TYPE_CHECKING:
    from hyperadvanced_ai.core.abstraction import HyperAIAbstraction
    from hyperadvanced_ai.core.base_module import BaseAIModule
    from hyperadvanced_ai.core.types import (
        AIModule,
        ModuleMetadata,
        ModuleInitializationError,
        T
    )

# Lazy loading for runtime
def __getattr__(name: str) -> Any:  # type: ignore
    if name in {'AIModule', 'ModuleMetadata', 'ModuleInitializationError', 'HyperAIAbstraction'}:
        from hyperadvanced_ai.core import abstraction
        return getattr(abstraction, name)
    if name == 'BaseAIModule':
        from hyperadvanced_ai.core import base_module
        return base_module.BaseAIModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
