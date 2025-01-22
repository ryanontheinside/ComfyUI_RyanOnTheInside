"""
Tooltip management system for ComfyUI nodes.
"""

from .tooltip_manager import TooltipManager, apply_tooltips
from .categories import register_all_tooltips

__all__ = ['TooltipManager', 'apply_tooltips']

# Register tooltips when the module is imported
register_all_tooltips()
