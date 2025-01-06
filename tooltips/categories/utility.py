"""Tooltips for utility-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for utility nodes"""

    # ImageIntervalSelect tooltips (inherits from: UtilityNode)
    TooltipManager.register_tooltips("ImageIntervalSelect", {
        "image": "Input image sequence (IMAGE type)",
        "interval": "Number of frames to skip between selections (1 to 100000)",
        "start_at": "Frame number to start selection from (0 to 100000)",
        "end_at": "Frame number to end selection at (0 to 100000, 0 means until end)"
    }, inherits_from='UtilityNode', description="Select frames from an image sequence at regular intervals, useful for reducing frame count or creating time-lapse effects.")

    # ImageIntervalSelectPercentage tooltips (inherits from: UtilityNode)
    TooltipManager.register_tooltips("ImageIntervalSelectPercentage", {
        "image": "Input image sequence (IMAGE type)",
        "interval_percentage": "Percentage of total frames to skip between selections (1 to 100)",
        "start_percentage": "Percentage point to start selection from (0 to 100)",
        "end_percentage": "Percentage point to end selection at (0 to 100)"
    }, inherits_from='UtilityNode', description="Select frames using percentages instead of frame numbers, making it easier to work with sequences of different lengths.")

    # ImageChunks tooltips (inherits from: UtilityNode)
    TooltipManager.register_tooltips("ImageChunks", {
        "image": "Input image sequence to arrange in a grid (IMAGE type)"
    }, inherits_from='UtilityNode', description="Arrange an image sequence into a grid layout, useful for creating sprite sheets or thumbnail previews.")

    # VideoChunks tooltips (inherits from: UtilityNode)
    TooltipManager.register_tooltips("VideoChunks", {
        "image": "Input video frames to arrange in grids (IMAGE type)",
        "chunk_size": "Number of frames per grid (minimum: 1)"
    }, inherits_from='UtilityNode', description="Create multiple grid layouts from a video sequence, with control over how many frames appear in each grid.")

    # Image_Shuffle tooltips (inherits from: UtilityNode)
    TooltipManager.register_tooltips("Image_Shuffle", {
        "image": "Input image sequence to shuffle (IMAGE type)",
        "shuffle_size": "Size of groups to shuffle together (minimum: 1)"
    }, inherits_from='UtilityNode', description="Randomly reorder frames in an image sequence, with control over group size for maintaining some temporal coherence.")

    # ImageDifference tooltips (inherits from: UtilityNode)
    TooltipManager.register_tooltips("ImageDifference", {
        "image": "Input image sequence to compute differences between frames (IMAGE type, minimum 2 frames)"
    }, inherits_from='UtilityNode', description="Calculate the visual difference between consecutive frames, useful for motion detection or transition effects.")

    # SwapDevice tooltips (inherits from: UtilityNode)
    TooltipManager.register_tooltips("SwapDevice", {
        "device": "Target device to move tensors to ('cpu' or 'cuda')",
        "image": "Optional input image to move to target device (IMAGE type)",
        "mask": "Optional input mask to move to target device (MASK type)"
    }, inherits_from='UtilityNode', description="Move images and masks between CPU and GPU memory to optimize processing speed and memory usage.")
