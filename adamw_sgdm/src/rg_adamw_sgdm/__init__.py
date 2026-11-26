"""Portable AdamW with an SGD-momentum magnitude graft."""

from .groups import parameter_groups
from .optimizer import AdamWSGDM

__all__ = ["AdamWSGDM", "parameter_groups"]
__version__ = "0.1.0"
