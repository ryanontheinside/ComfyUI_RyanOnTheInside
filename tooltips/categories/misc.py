"""Tooltips for misc-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for misc nodes"""

    # WhisperToPromptTravel tooltips
    TooltipManager.register_tooltips("WhisperToPromptTravel", {
        "segments_alignment": "JSON string containing segment alignments from Whisper. Each segment should have 'start' (timestamp) and 'value' (text) fields.",
        "fps": "Frame rate of the video to sync with (0.1 to 120.0 fps). Used to convert timestamps to frame numbers.",
    })
