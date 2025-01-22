import os
from ... import RyanOnTheInside
from ...tooltips import apply_tooltips
from .flex_externals import FlexExternalModulator
import torch
import numpy as np

# Get paths
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
custom_nodes_dir = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))

# Import Advanced-ControlNet components
adv_controlnet_dir = os.path.join(custom_nodes_dir, 'ComfyUI-Advanced-ControlNet')
adv_controlnet_utils_path = os.path.join(adv_controlnet_dir, 'adv_control', 'utils.py')
adv_controlnet_nodes_path = os.path.join(adv_controlnet_dir, 'adv_control', 'nodes_keyframes.py')

# Import the modules using the helper from FlexExternalModulator
acn_utils = FlexExternalModulator.import_module_from_path('acn_utils', adv_controlnet_utils_path)
acn_nodes = FlexExternalModulator.import_module_from_path('acn_nodes', adv_controlnet_nodes_path)

# Get required classes
TimestepKeyframe = acn_utils.TimestepKeyframe
TimestepKeyframeGroup = acn_utils.TimestepKeyframeGroup
LatentKeyframe = acn_utils.LatentKeyframe
LatentKeyframeGroup = acn_utils.LatentKeyframeGroup
SI = acn_utils.StrengthInterpolation

_acn_category = f"{FlexExternalModulator.CATEGORY}/Advanced-ControlNet"

@apply_tooltips
class FeatureToTimestepKeyframe(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature": ("FEATURE",),
                "select_every": ("INT", {"default": 1, "min": 1, "max": 999}),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": 10000}),
                "inherit_missing": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "prev_timestep_kf": ("TIMESTEP_KEYFRAME",),
                "cn_weights": ("CONTROL_NET_WEIGHTS",),
                "latent_keyframe": ("LATENT_KEYFRAME",),
            }
        }
    
    RETURN_TYPES = ("TIMESTEP_KEYFRAME",)
    FUNCTION = "convert"
    CATEGORY = _acn_category

    def convert(self, feature, select_every, guarantee_steps, inherit_missing,
                prev_timestep_kf=None, cn_weights=None, latent_keyframe=None):
        # Create or clone keyframe group
        if prev_timestep_kf is None:
            keyframe_group = TimestepKeyframeGroup()
        else:
            keyframe_group = prev_timestep_kf.clone()

        # Get number of frames from feature
        num_frames = feature.frame_count

        # Create keyframes for every nth frame based on select_every
        for i in range(0, num_frames, select_every):
            # Calculate percentage through animation for this frame
            start_percent = i / (num_frames - 1) if num_frames > 1 else 0.0
            
            # Get strength value for this frame
            strength = float(feature.get_value_at_frame(i))

            # Create keyframe with the values
            keyframe = TimestepKeyframe(
                start_percent=start_percent,
                strength=strength,
                control_weights=cn_weights,
                latent_keyframes=latent_keyframe,
                inherit_missing=inherit_missing,
                guarantee_steps=guarantee_steps
            )
            keyframe_group.add(keyframe)

        return (keyframe_group,)

@apply_tooltips
class FeatureToLatentKeyframe(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature": ("FEATURE",),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME",),
            }
        }
    
    RETURN_TYPES = ("LATENT_KEYFRAME",)
    FUNCTION = "convert"
    CATEGORY = _acn_category

    def convert(self, feature, prev_latent_kf=None):
        # Create or clone keyframe group
        if prev_latent_kf is None:
            keyframe_group = LatentKeyframeGroup()
        else:
            keyframe_group = prev_latent_kf.clone()

        # For each frame in the feature sequence
        for frame_idx in range(feature.frame_count):
            # Get strength value for this frame
            strength = float(feature.get_value_at_frame(frame_idx))

            # Create keyframe for this frame's strength value
            # Here frame_idx correctly maps to batch_index for animation
            keyframe = LatentKeyframe(frame_idx, strength)
            keyframe_group.add(keyframe)

        return (keyframe_group,)
