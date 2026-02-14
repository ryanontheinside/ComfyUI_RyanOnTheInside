import torch
import comfy.model_management
import numpy as np
from .nodes.latents.flex_latent_base import FlexLatentBase
from .tooltips import apply_tooltips, TooltipManager

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
        if not isinstance(latent1, dict) or not isinstance(latent2, dict):
            raise ValueError("Inputs must be latent dictionaries")

        if "samples" not in latent1 or "samples" not in latent2:
            raise ValueError("Inputs must contain 'samples' key")

        s1 = latent1["samples"]
        s2 = latent2["samples"]

        samples_out = latent1.copy()

        # Ensure both latents have the same shape
        if s1.shape != s2.shape:
            # Match time dimension (last dim) via interpolation
            if s1.dim() == 3:
                # ACE-Step 1.5: [B, C, T]
                s2 = torch.nn.functional.interpolate(s2, size=s1.shape[2], mode='linear', align_corners=False)
            else:
                # ACE-Step 1.0: [B, C, H, T]
                s2 = torch.nn.functional.interpolate(s2, size=s1.shape[2:], mode='bilinear', align_corners=False)

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


#NOTE: SPIKE, WIP
class FlexlatentAudioBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "latent1": ("LATENT",),
            "latent2": ("LATENT",),
            "blend_factors": ("FLOAT", {
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "round": 0.01,
                "forceInput": True,
                "display": "numberlist"
            }),
            "blend_mode": (["normal", "add", "subtract", "multiply", "overlay"], {"default": "normal"}),
            "interpolation": (["linear", "step"], {"default": "linear"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blend"
    CATEGORY = "latent/audio"

    def blend(self, latent1, latent2, blend_factors, blend_mode: str, interpolation: str):
        if not isinstance(latent1, dict) or not isinstance(latent2, dict):
            raise ValueError("Inputs must be latent dictionaries")

        if "samples" not in latent1 or "samples" not in latent2:
            raise ValueError("Inputs must contain 'samples' key")

        s1 = latent1["samples"]
        s2 = latent2["samples"]
        is_3d = s1.dim() == 3  # ACE-Step 1.5: [B, C, T] vs 1.0: [B, C, H, T]

        samples_out = latent1.copy()

        # Ensure both latents have the same shape
        if s1.shape != s2.shape:
            if is_3d:
                s2 = torch.nn.functional.interpolate(s2, size=s1.shape[2], mode='linear', align_corners=False)
            else:
                s2 = torch.nn.functional.interpolate(s2, size=s1.shape[2:], mode='bilinear', align_corners=False)

        # Convert blend_factors to a list if it's a single float
        if isinstance(blend_factors, float):
            blend_factors_list = [blend_factors]
        else:
            blend_factors_list = blend_factors

        # Clamp values to 0.0-1.0 range
        blend_factors_list = [max(0.0, min(1.0, f)) for f in blend_factors_list]

        if len(blend_factors_list) == 0:
            raise ValueError("No blend factors provided")

        # Temporal dimension is last: shape[2] for 3D, shape[3] for 4D
        temporal_length = s1.shape[-1]

        # Create blend factor tensor across temporal dimension
        if len(blend_factors_list) == 1:
            blend_factor_tensor = torch.ones(temporal_length, device=s1.device) * blend_factors_list[0]
        else:
            points = torch.linspace(0, temporal_length-1, len(blend_factors_list))
            values = torch.tensor(blend_factors_list, device=s1.device)
            positions = torch.arange(temporal_length, device=s1.device)

            if interpolation == "step":
                indices = torch.searchsorted(points, positions) - 1
                indices = torch.clamp(indices, 0, len(blend_factors_list) - 1)
                blend_factor_tensor = values[indices]
            else:  # linear interpolation
                blend_factor_tensor = torch.zeros(temporal_length, device=s1.device)

                for i in range(len(points) - 1):
                    start, end = points[i], points[i+1]
                    mask = (positions >= start) & (positions <= end)
                    weight = (positions[mask] - start) / (end - start)
                    blend_factor_tensor[mask] = values[i] * (1 - weight) + values[i+1] * weight

                # Fill edges
                blend_factor_tensor[positions < points[0]] = values[0]
                blend_factor_tensor[positions > points[-1]] = values[-1]

        # Ensure blend_factor_tensor has the correct temporal length
        if blend_factor_tensor.shape[0] != temporal_length:
            blend_factor_tensor = torch.nn.functional.interpolate(
                blend_factor_tensor.unsqueeze(0).unsqueeze(0),
                size=temporal_length,
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)

        # Expand blend_factor_tensor to match latent dimensions
        # 3D [B, C, T]: unsqueeze to [1, 1, T]
        # 4D [B, C, H, T]: unsqueeze to [1, 1, 1, T]
        if is_3d:
            blend_factor_tensor = blend_factor_tensor.unsqueeze(0).unsqueeze(0)
        else:
            blend_factor_tensor = blend_factor_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        target_shape = list(s1.shape)
        
        # Apply blend mode with time-varying factors
        if blend_mode == "normal":
            blend_factor_expanded = blend_factor_tensor.expand(target_shape)
            blended = s1 * blend_factor_expanded + s2 * (1 - blend_factor_expanded)
        elif blend_mode == "add":
            blend_factor_expanded = blend_factor_tensor.expand(target_shape)
            blended = s1 + s2 * blend_factor_expanded
        elif blend_mode == "subtract":
            blend_factor_expanded = blend_factor_tensor.expand(target_shape)
            blended = s1 - s2 * blend_factor_expanded
        elif blend_mode == "multiply":
            blend_factor_expanded = blend_factor_tensor.expand(target_shape)
            blended = s1 * (1 - blend_factor_expanded + s2 * blend_factor_expanded)
        elif blend_mode == "overlay":
            # Overlay blend mode: combines multiply and screen blend modes
            blend_factor_expanded = blend_factor_tensor.expand(target_shape)
            mask = s1 > 0.5
            blended = torch.zeros_like(s1)
            blended[mask] = 1 - (1 - s1[mask]) * (1 - s2[mask] * blend_factor_expanded[mask])
            blended[~mask] = s1[~mask] * (1 - blend_factor_expanded[~mask] + s2[~mask] * blend_factor_expanded[~mask])
        else:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")

        # Normalize the output to prevent extreme values
        blended = torch.clamp(blended, -1.0, 1.0)
        
        samples_out["samples"] = blended
        samples_out["type"] = "audio"  # Ensure output is marked as audio
        return (samples_out,)


NODE_CLASS_MAPPINGS = {
    "AudioLatentBlend": AudioLatentBlend,
    "FlexlatentAudioBlend": FlexlatentAudioBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioLatentBlend": "Audio Latent Blend (DEPRECATED USE FLEX LATENT BLEND)",
    "FlexlatentAudioBlend": "FlexlatentAudioBlend (DEPRECATED USE FLEX LATENT BLEND)",
} 