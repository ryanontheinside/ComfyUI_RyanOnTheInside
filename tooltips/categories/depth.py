"""Tooltips for depth-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for depth nodes"""

    # FlexDepthBase tooltips (inherits from: RyanOnTheInside, FlexBase)
    TooltipManager.register_tooltips("FlexDepthBase", {
        "depth_maps": "Input depth maps to be processed",
        "strength": "Overall strength of the effect (0.0 to 1.0)",
        "feature_threshold": "Minimum feature value to apply the effect (0.0 to 1.0)",
        "feature_param": "Parameter to be modulated by the feature",
        "feature_mode": "How the feature modulates the parameter ('relative' or 'absolute')",
        "opt_feature": "Optional feature input for parameter modulation",
        "opt_feature_pipe": "Optional feature pipe for frame synchronization"
    }, inherits_from=['RyanOnTheInside', 'FlexBase'])

    # DepthInjection tooltips (inherits from: FlexDepthBase)
    TooltipManager.register_tooltips("DepthInjection", {
        "mask": "Input mask defining areas for depth modification",
        "gradient_steepness": "Controls the steepness of the spherical gradient (0.1 to 10.0)",
        "depth_min": "Minimum depth value for the modified areas (0.0 to 1.0)",
        "depth_max": "Maximum depth value for the modified areas (0.0 to 1.0)"
    }, inherits_from='FlexDepthBase')

    # DepthBlender tooltips (inherits from: FlexDepthBase)
    TooltipManager.register_tooltips("DepthBlender", {
        "other_depth_maps": "Secondary depth maps to blend with",
        "blend_mode": "Method of blending ('add', 'subtract', 'multiply', 'average')"
    }, inherits_from='FlexDepthBase')

    # DepthRippleEffect tooltips (inherits from: FlexDepthBase)
    TooltipManager.register_tooltips("DepthRippleEffect", {
        "ripple_amplitude": "Amplitude of the ripple effect (0.0 to 0.5)",
        "ripple_frequency": "Frequency of the ripples (1.0 to 100.0)",
        "ripple_phase": "Phase offset of the ripples (0.0 to 2π)",
        "curvature": "Interpolation between linear and circular patterns (0.0 to 1.0)"
    }, inherits_from='FlexDepthBase')
