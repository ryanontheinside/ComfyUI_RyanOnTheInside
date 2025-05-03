import os

def get_case_insensitive_path(base_dir, target_name):
    """Find a directory path case-insensitively within base_dir."""
    if not os.path.exists(base_dir):
        return None
    for name in os.listdir(base_dir):
        if name.lower() == target_name.lower():
            return os.path.join(base_dir, name)
    return None

# Variables to track the availability of external modules
parent_dir = os.path.dirname(os.path.dirname(__file__))

alp_path = get_case_insensitive_path(parent_dir, "ComfyUI-AdvancedLivePortrait")
HAS_ADVANCED_LIVE_PORTRAIT = alp_path is not None
print(f"Checking for AdvancedLivePortrait at: {alp_path}")

acn_path = get_case_insensitive_path(parent_dir, "ComfyUI-Advanced-ControlNet")
HAS_ADVANCED_CONTROLNET = acn_path is not None
print(f"Checking for Advanced-ControlNet at: {acn_path}")

ad_path = get_case_insensitive_path(parent_dir, "ComfyUI-AnimateDiff-Evolved")
HAS_ANIMATEDIFF = ad_path is not None
print(f"Checking for AnimateDiff-Evolved at: {ad_path}")

# Conditional imports for AdvancedLivePortrait
if HAS_ADVANCED_LIVE_PORTRAIT:
    try:
        from .nodes.flex.flex_externals_advanced_live_portrait import FlexExpressionEditor
    except Exception as e:
        print(f"Error loading AdvancedLivePortrait nodes: {str(e)}")
        HAS_ADVANCED_LIVE_PORTRAIT = False
else:
    print(
        "ComfyUI-AdvancedLivePortrait not found. "
        "FlexExpressionEditor will not be available. "
        "Install ComfyUI-AdvancedLivePortrait and restart ComfyUI."
    )

# Conditional imports for Advanced-ControlNet
if HAS_ADVANCED_CONTROLNET:
    try:
        from .nodes.flex.flex_externals_advanced_controlnet import (
            FeatureToLatentKeyframe,
            #WIP
            # FeatureToTimestepKeyframe,
        )
    except Exception as e:
        print(f"Error loading Advanced-ControlNet feature nodes: {str(e)}")
        HAS_ADVANCED_CONTROLNET = False
else:
    print(
        "ComfyUI-Advanced-ControlNet not found. "
        "Advanced-ControlNet feature nodes will not be available. "
        "Install ComfyUI-Advanced-ControlNet and restart ComfyUI."
    )

# Conditional imports for AnimateDiff
if HAS_ANIMATEDIFF:
    try:
        from .nodes.flex.flex_externals_animatediff import (
            FeatureToADKeyframe,
            #WIP
            # FeatureToCameraKeyframe,
            # FeatureToPIAKeyframe,
        )
    except Exception as e:
        print(f"Error loading AnimateDiff feature nodes: {str(e)}")
        HAS_ANIMATEDIFF = False
else:
    print(
        "ComfyUI-AnimateDiff-Evolved not found. "
        "AnimateDiff feature nodes will not be available. "
        "Install ComfyUI-AnimateDiff-Evolved and restart ComfyUI."
    )

# Prepare a dictionary to hold the NODE_CLASS_MAPPINGS additions
EXTERNAL_NODE_CLASS_MAPPINGS = {}

# Update the mapping based on available integrations
if HAS_ADVANCED_LIVE_PORTRAIT:
    EXTERNAL_NODE_CLASS_MAPPINGS["FlexExpressionEditor"] = FlexExpressionEditor

if HAS_ADVANCED_CONTROLNET:
    EXTERNAL_NODE_CLASS_MAPPINGS.update({
        # "FeatureToTimestepKeyframe": FeatureToTimestepKeyframe,
        "FeatureToLatentKeyframe": FeatureToLatentKeyframe,
    })

if HAS_ANIMATEDIFF:
    EXTERNAL_NODE_CLASS_MAPPINGS.update({
        "FeatureToADKeyframe": FeatureToADKeyframe,
        # "FeatureToCameraKeyframe": FeatureToCameraKeyframe,
        # "FeatureToPIAKeyframe": FeatureToPIAKeyframe,
    })