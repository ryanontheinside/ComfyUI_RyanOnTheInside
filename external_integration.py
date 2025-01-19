import os

# Variables to track the availability of external modules
HAS_ADVANCED_LIVE_PORTRAIT = os.path.exists(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "ComfyUI-AdvancedLivePortrait"
    )
)

HAS_ADVANCED_CONTROLNET = os.path.exists(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "ComfyUI-Advanced-ControlNet"
    )
)

#WIP
HAS_ANIMATEDIFF = os.path.exists(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "ComfyUI-AnimateDiff-Evolved"
    )
)

# HAS_ANIMATEDIFF=False

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
            FeatureToTimestepKeyframe,
            FeatureToLatentKeyframe,
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
            FeatureToCameraKeyframe,
            FeatureToPIAKeyframe,
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
        "FeatureToTimestepKeyframe": FeatureToTimestepKeyframe,
        "FeatureToLatentKeyframe": FeatureToLatentKeyframe,
    })

if HAS_ANIMATEDIFF:
    EXTERNAL_NODE_CLASS_MAPPINGS.update({
        "FeatureToADKeyframe": FeatureToADKeyframe,
        "FeatureToCameraKeyframe": FeatureToCameraKeyframe,
        "FeatureToPIAKeyframe": FeatureToPIAKeyframe,
    })