import os
from ... import RyanOnTheInside
from ...tooltips import apply_tooltips
from .flex_externals import FlexExternalModulator
import torch

#NOTE: work in progress

# Get paths
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
custom_nodes_dir = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))

# Import AnimateDiff components
animatediff_dir = os.path.join(custom_nodes_dir, 'ComfyUI-AnimateDiff-Evolved')
animatediff_nodes_path = os.path.join(animatediff_dir, 'animatediff', 'nodes_gen2.py')
animatediff_camera_path = os.path.join(animatediff_dir, 'animatediff', 'nodes_cameractrl.py')
animatediff_pia_path = os.path.join(animatediff_dir, 'animatediff', 'nodes_pia.py')

# Import the modules using the helper from FlexExternalModulator
ad_nodes = FlexExternalModulator.import_module_from_path('animatediff_nodes', animatediff_nodes_path)
ad_camera = FlexExternalModulator.import_module_from_path('animatediff_camera', animatediff_camera_path)
ad_pia = FlexExternalModulator.import_module_from_path('animatediff_pia', animatediff_pia_path)

# Get required classes
ADKeyframeGroup = ad_nodes.ADKeyframeGroup
ADKeyframe = ad_nodes.ADKeyframe
CAM = ad_camera.CAM
PIA_RANGES = ad_pia.PIA_RANGES
InputPIA_PaperPresets = ad_pia.InputPIA_PaperPresets

_ad_category = f"{FlexExternalModulator.CATEGORY}/AnimateDiff"

@apply_tooltips
class FeatureToADKeyframe(FlexExternalModulator):

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "select_every": ("INT", {"default": 1, "min": 1, "max": 999}),
            },
            "optional": {
                "scale_feature": ("FEATURE",),
                "effect_feature": ("FEATURE",),
            }
        }
    
    RETURN_TYPES = ("AD_KEYFRAMES",)
    FUNCTION = "convert"
    CATEGORY = _ad_category


    def convert(self, select_every: int, scale_feature=None, effect_feature=None):
        if scale_feature is None and effect_feature is None:
            raise ValueError("At least one of scale_feature or effect_feature must be provided")

        # Create new keyframe group
        keyframes = ADKeyframeGroup()

        # Get number of frames from whichever feature is provided
        if scale_feature is not None:
            num_frames = scale_feature.frame_count
        else:
            num_frames = effect_feature.frame_count

        # Create keyframes for every nth frame based on select_every
        for i in range(0, num_frames, select_every):
            # Calculate percentage through animation for this frame
            start_percent = i / (num_frames - 1) if num_frames > 1 else 0.0
            
            # Get values for this frame if features are provided
            scale_val = float(scale_feature.get_value_at_frame(i)) if scale_feature is not None else None
            effect_val = float(effect_feature.get_value_at_frame(i)) if effect_feature is not None else None

            # Create keyframe with the values
            keyframe = ADKeyframe(
                start_percent=start_percent,
                scale_multival=scale_val,
                effect_multival=effect_val
            )
            keyframes.add(keyframe)

        return (keyframes,)

@apply_tooltips
class FeatureToCameraKeyframe(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature": ("FEATURE",),
                "camera_motion": (CAM._LIST,),
                "select_every": ("INT", {"default": 1, "min": 1, "max": 999}),
                "inherit_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AD_KEYFRAMES",)
    FUNCTION = "convert"
    CATEGORY = _ad_category

    def convert(self, feature, camera_motion, select_every, inherit_missing):
        # Create a new keyframe group
        keyframe_group = ADKeyframeGroup()
        
        # Create keyframes for every nth frame based on select_every
        for i in range(0, feature.frame_count, select_every):
            # Calculate percentage through animation for this frame
            start_percent = i / (feature.frame_count - 1) if feature.frame_count > 1 else 0.0
            
            # Get camera value for this frame
            camera_val = float(feature.get_value_at_frame(i))
            
            # Create keyframe with camera control value
            keyframe = ADKeyframe(
                start_percent=start_percent,
                cameractrl_multival=camera_val,  # Single value for this frame
                inherit_missing=inherit_missing
            )
            
            keyframe_group.add(keyframe)
            
        return (keyframe_group,)

@apply_tooltips
class FeatureToPIAKeyframe(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature": ("FEATURE",),
                "preset": (PIA_RANGES._LIST_ALL,),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "inherit_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AD_KEYFRAMES",)
    FUNCTION = "convert"
    CATEGORY = _ad_category

    def convert(self, feature, preset, start_percent, guarantee_steps, inherit_missing):
        # Create a new keyframe group
        keyframe_group = ADKeyframeGroup()
        
        # Create PIA input with the feature values
        pia_input = InputPIA_PaperPresets(
            preset=preset,
            index=0,  # Will be determined by frame index
            mult_multival=[feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        )
        
        # Create keyframe with PIA settings
        keyframe = ADKeyframe(
            start_percent=start_percent,
            pia_input=pia_input,
            inherit_missing=inherit_missing,
            guarantee_steps=guarantee_steps
        )
        
        keyframe_group.add(keyframe)
        return (keyframe_group,) 