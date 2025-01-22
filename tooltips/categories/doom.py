"""Tooltips for doom-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for doom nodes"""

    # Doom tooltips
    TooltipManager.register_tooltips("Doom", {
        # TODO: Add parameter tooltips
    }, description="It's Doom.")
