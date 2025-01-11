import os
from ... import RyanOnTheInside
from ...tooltips import apply_tooltips
from .flex_externals import FlexExternalModulator

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

class FlexExternalModulator(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/ExternalTargets"

@apply_tooltips
class FeatureToADKeyframe(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature": ("FEATURE",),
                "scale_feature": ("FEATURE",),
                "effect_feature": ("FEATURE",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "inherit_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AD_KEYFRAMES",)
    FUNCTION = "convert"

    def convert(self, feature, scale_feature, effect_feature, start_percent, guarantee_steps, inherit_missing):
        # Create a new keyframe group
        keyframe_group = ADKeyframeGroup()
        
        # Convert features to multivals
        scale_values = [scale_feature.get_value_at_frame(i) for i in range(scale_feature.frame_count)]
        effect_values = [effect_feature.get_value_at_frame(i) for i in range(effect_feature.frame_count)]
        
        # Create keyframe with the feature values
        keyframe = ADKeyframe(
            start_percent=start_percent,
            scale_multival=scale_values,
            effect_multival=effect_values,
            inherit_missing=inherit_missing,
            guarantee_steps=guarantee_steps
        )
        
        keyframe_group.add(keyframe)
        return (keyframe_group,)

@apply_tooltips
class FeatureToCameraKeyframe(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature": ("FEATURE",),
                "camera_motion": (CAM._LIST,),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "inherit_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AD_KEYFRAMES",)
    FUNCTION = "convert"

    def convert(self, feature, camera_motion, start_percent, guarantee_steps, inherit_missing):
        # Create a new keyframe group
        keyframe_group = ADKeyframeGroup()
        
        # Get camera motion values for each frame
        camera_values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Create keyframe with camera control values
        keyframe = ADKeyframe(
            start_percent=start_percent,
            cameractrl_multival=camera_values,  # This will control the strength of the camera motion
            inherit_missing=inherit_missing,
            guarantee_steps=guarantee_steps
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