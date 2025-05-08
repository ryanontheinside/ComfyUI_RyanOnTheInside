import torch
import comfy.model_management
import numpy as np
from .nodes.latents.flex_latent_base import FlexLatentBase
from .tooltips import apply_tooltips

class AudioLatentBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent1": ("LATENT",),
            "latent2": ("LATENT",),
            "blend_factor": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            }),
            "blend_mode": (["normal", "add", "subtract", "multiply", "overlay"], {"default": "normal"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blend"
    CATEGORY = "latent/audio"

    def blend(self, latent1, latent2, blend_factor: float, blend_mode: str):
        # Ensure we're working with latent dictionaries
        if not isinstance(latent1, dict) or not isinstance(latent2, dict):
            raise ValueError("Inputs must be latent dictionaries")
            
        if "samples" not in latent1 or "samples" not in latent2:
            raise ValueError("Inputs must contain 'samples' key")

        # Check if the latents have the correct shape for audio (8 channels)
        s1 = latent1["samples"]
        s2 = latent2["samples"]
        
        if s1.shape[1] != 8 or s2.shape[1] != 8:
            raise ValueError("Both inputs must be audio latents (8 channels)")

        samples_out = latent1.copy()
        
        # Ensure both latents have the same shape
        if s1.shape != s2.shape:
            # Reshape s2 to match s1's dimensions
            s2 = torch.nn.functional.interpolate(
                s2, 
                size=s1.shape[2:],
                mode='linear',
                align_corners=False
            )

        # Apply blend mode
        if blend_mode == "normal":
            blended = s1 * blend_factor + s2 * (1 - blend_factor)
        elif blend_mode == "add":
            blended = s1 + s2 * blend_factor
        elif blend_mode == "subtract":
            blended = s1 - s2 * blend_factor
        elif blend_mode == "multiply":
            blended = s1 * (1 - blend_factor + s2 * blend_factor)
        elif blend_mode == "overlay":
            # Overlay blend mode: combines multiply and screen blend modes
            mask = s1 > 0.5
            blended = torch.zeros_like(s1)
            blended[mask] = 1 - (1 - s1[mask]) * (1 - s2[mask] * blend_factor)
            blended[~mask] = s1[~mask] * (1 - blend_factor + s2[~mask] * blend_factor)
        else:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")

        # Normalize the output to prevent extreme values
        blended = torch.clamp(blended, -1.0, 1.0)
        
        samples_out["samples"] = blended
        samples_out["type"] = "audio"  # Ensure output is marked as audio
        return (samples_out,)

@apply_tooltips
class FlexAudioLatentBlend(FlexLatentBase):

    #NOTE: WIP
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"]["feature_param"] = cls.get_modifiable_params()
        inputs["required"].update({
            "latent_2": ("LATENT",),
            "blend_mode": (["normal", "add", "subtract", "multiply", "overlay"], {"default": "normal"}),
            "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["blend_factor", "None"]

    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/Audio"
    
    def apply_effect_internal(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        feature_value = kwargs['feature_value']
        strength = kwargs['strength']
        feature_param = kwargs['feature_param']
        feature_mode = kwargs['feature_mode']
        latent_2 = kwargs['latent_2']
        blend_mode = kwargs['blend_mode']
        blend_factor = kwargs['blend_factor']
        frame_index = kwargs['frame_index']

        # Modulate the blend_factor parameter if selected
        if feature_param == "blend_factor":
            blend_factor = self.modulate_param(
                "blend_factor",
                blend_factor,
                feature_value,
                strength,
                feature_mode
            )
            # Ensure blend_factor remains within [0, 1]
            blend_factor = np.clip(blend_factor, 0.0, 1.0)

        # Get the corresponding frame from latent_2
        latent_2_np = latent_2["samples"].cpu().numpy()[frame_index % latent_2["samples"].shape[0]]
        
        # Check if the latents have the correct shape for audio (8 channels)
        if latent.shape[0] != 8 or latent_2_np.shape[0] != 8:
            raise ValueError("Both inputs must be audio latents (8 channels)")

        # Ensure both latents have the same shape
        if latent.shape != latent_2_np.shape:
            # We can't do interpolation in numpy easily, so we'll resize in PyTorch later
            pass
        
        # Apply blend mode
        if blend_mode == "normal":
            result = latent * blend_factor + latent_2_np * (1 - blend_factor)
        elif blend_mode == "add":
            result = latent + latent_2_np * blend_factor
        elif blend_mode == "subtract":
            result = latent - latent_2_np * blend_factor
        elif blend_mode == "multiply":
            result = latent * (1 - blend_factor + latent_2_np * blend_factor)
        elif blend_mode == "overlay":
            # Overlay blend mode: combines multiply and screen blend modes
            mask = latent > 0.5
            result = np.zeros_like(latent)
            result[mask] = 1 - (1 - latent[mask]) * (1 - latent_2_np[mask] * blend_factor)
            result[~mask] = latent[~mask] * (1 - blend_factor + latent_2_np[~mask] * blend_factor)
        else:
            # Default to normal blend
            result = latent * blend_factor + latent_2_np * (1 - blend_factor)

        # Normalize the output to prevent extreme values
        result = np.clip(result, -1.0, 1.0)
        return result

NODE_CLASS_MAPPINGS = {
    "AudioLatentBlend": AudioLatentBlend,
    "FlexAudioLatentBlend": FlexAudioLatentBlend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioLatentBlend": "Audio Latent Blend",
    "FlexAudioLatentBlend": "Flex Audio Latent Blend"
} 