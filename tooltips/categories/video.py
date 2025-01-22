"""Tooltips for video-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for video nodes"""

    # FlexVideoBase tooltips (inherits from: FlexBase, ABC)
    TooltipManager.register_tooltips("FlexVideoBase", {
        "images": "Input video frames (IMAGE type)",
        "feature": "Feature used to modulate the effect (FEATURE type)",
        "strength": "Overall strength of the effect (0.0 to 2.0)",
        "feature_mode": "How the feature modulates the parameter ('relative' or 'absolute')",
        "feature_param": "Parameter to be modulated by the feature",
        "feature_threshold": "Minimum feature value to apply the effect (0.0 to 1.0)",
        "feature_pipe": "Feature pipe containing frame information (FEATURE_PIPE type)"
    }, inherits_from=['FlexBase', 'ABC'])

    # FlexVideoDirection tooltips (inherits from: FlexVideoBase)
    TooltipManager.register_tooltips("FlexVideoDirection", {
        "feature_pipe": "Feature pipe containing frame information (FEATURE_PIPE type)",
        "feature_values": "Array of feature values for each frame (0.0 to 1.0)"
    }, inherits_from='FlexVideoBase')

    # FlexVideoSeek tooltips (inherits from: FlexVideoBase)
    TooltipManager.register_tooltips("FlexVideoSeek", {
        "feature_mode": "Only supports 'relative' mode",
        "reverse": "Whether to play the video in reverse",
        "feature_values": "Array of feature values for each frame (0.0 to 1.0)"
    }, inherits_from='FlexVideoBase')

    # FlexVideoFrameBlend tooltips (inherits from: FlexVideoBase)
    TooltipManager.register_tooltips("FlexVideoFrameBlend", {
        "blend_strength": "Strength of the frame blending effect (0.0 to 1.0)",
        "frame_offset_ratio": "Ratio of frame offset for blending (0.0 to 1.0)",
        "direction_bias": "Bias for blending direction (0.0 to 1.0)",
        "blend_mode": "Mode for frame blending ('normal', 'additive', 'multiply', 'screen')",
        "motion_blur_strength": "Strength of motion blur effect (0.0 to 1.0)"
    }, inherits_from='FlexVideoBase')

    # FlexVideoSpeed tooltips (inherits from: FlexVideoBase)
    TooltipManager.register_tooltips("FlexVideoSpeed", {
        "speed_factor": "Speed factor for playback (-100.0 to 100.0)",
        "interpolation_mode": "Method for frame interpolation ('none', 'linear', 'Farneback', 'rife47', 'rife49')",
        "fast_mode": "Enable fast processing mode for RIFE interpolation",
        "ensemble": "Enable ensemble mode for better quality in RIFE interpolation",
        "scale_factor": "Scale factor for processing (0.25, 0.5, 1.0, 2.0, 4.0)"
    }, inherits_from='FlexVideoBase')
