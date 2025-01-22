"""Tooltips for preprocessors-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for preprocessors nodes"""

    # PoseInterpolator tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("PoseInterpolator", {
        "pose_1": "First pose keypoint sequence to interpolate between",
        "pose_2": "Second pose keypoint sequence to interpolate between",
        "feature": "Feature that controls the interpolation amount between poses",
        "strength": "Overall strength of the interpolation effect (0.0 to 1.0)",
        "interpolation_mode": "Method for interpolating between poses: Linear (straight interpolation) or Spherical (curved path interpolation)",
        "omit_missing_points": "When enabled, missing keypoints in either pose will be set to zero in the output. When disabled, uses the available keypoint from either pose.",
    }, inherits_from='RyanOnTheInside')
