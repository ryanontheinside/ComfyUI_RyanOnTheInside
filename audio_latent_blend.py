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
        # Ensure we're working with latent dictionaries
        if not isinstance(latent1, dict) or not isinstance(latent2, dict):
            raise ValueError("Inputs must be latent dictionaries")
            
        if "samples" not in latent1 or "samples" not in latent2:
            raise ValueError("Inputs must contain 'samples' key")

        # Check if the latents have the correct shape for audio (8 channels)
        s1 = latent1["samples"]
        s2 = latent2["samples"]
        
        print(f"Input latent shapes - s1: {s1.shape}, s2: {s2.shape}")
        
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
            print(f"Resized s2 shape: {s2.shape}")
        
        # Convert blend_factors to a list if it's a single float
        if isinstance(blend_factors, float):
            blend_factors_list = [blend_factors]
        else:
            # Already a list of floats
            blend_factors_list = blend_factors
            
        # Clamp values to 0.0-1.0 range
        blend_factors_list = [max(0.0, min(1.0, f)) for f in blend_factors_list]
        
        print(f"Blend factors: {blend_factors_list}")
        
        if len(blend_factors_list) == 0:
            raise ValueError("No blend factors provided")
            
        # Get the temporal dimension size
        temporal_length = s1.shape[3]
        print(f"Temporal length: {temporal_length}")
        
        # Create blend factor tensor across temporal dimension
        if len(blend_factors_list) == 1:
            # If only one factor, use it uniformly
            blend_factor_tensor = torch.ones(temporal_length, device=s1.device) * blend_factors_list[0]
            print(f"Using single blend factor: {blend_factors_list[0]}")
        else:
            # Create interpolation points
            points = torch.linspace(0, temporal_length-1, len(blend_factors_list))
            values = torch.tensor(blend_factors_list, device=s1.device)
            
            print(f"Interpolation points: {points.tolist()}")
            print(f"Values at points: {values.tolist()}")
            
            # Create a tensor for all temporal positions
            positions = torch.arange(temporal_length, device=s1.device)
            
            if interpolation == "step":
                # For step interpolation, find nearest point
                indices = torch.searchsorted(points, positions) - 1
                indices = torch.clamp(indices, 0, len(blend_factors_list) - 1)
                blend_factor_tensor = values[indices]
                print(f"Using step interpolation")
            else:  # linear interpolation
                # For positions before first point
                blend_factor_tensor = torch.zeros(temporal_length, device=s1.device)
                
                # For each segment between points
                for i in range(len(points) - 1):
                    # Find positions in this segment
                    start, end = points[i], points[i+1]
                    mask = (positions >= start) & (positions <= end)
                    
                    # Calculate interpolation weight
                    weight = (positions[mask] - start) / (end - start)
                    
                    # Interpolate values
                    blend_factor_tensor[mask] = values[i] * (1 - weight) + values[i+1] * weight
                print(f"Using linear interpolation")
                
                # For positions before first point, use first value
                mask = positions < points[0]
                blend_factor_tensor[mask] = values[0]
                
                # For positions after last point, use last value
                mask = positions > points[-1]
                blend_factor_tensor[mask] = values[-1]
        
        # Print blend factor tensor stats
        print(f"Blend factor tensor stats - min: {blend_factor_tensor.min().item()}, max: {blend_factor_tensor.max().item()}, mean: {blend_factor_tensor.mean().item()}")
        print(f"First few blend factors: {blend_factor_tensor[:5].tolist()}")
        print(f"Last few blend factors: {blend_factor_tensor[-5:].tolist()}")
        
        # Ensure blend_factor_tensor has the shape [temporal_length]
        if blend_factor_tensor.shape[0] != temporal_length:
            blend_factor_tensor = torch.nn.functional.interpolate(
                blend_factor_tensor.unsqueeze(0).unsqueeze(0),
                size=temporal_length,
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            print(f"Resized blend factor tensor shape: {blend_factor_tensor.shape}")
            
        # Expand blend_factor_tensor to match latent dimensions [batch, channels, 16, time]
        # First create the expanded shape manually to ensure compatibility
        batch_size, num_channels = s1.shape[0], s1.shape[1]
        blend_factor_tensor = blend_factor_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, temporal_length]
        
        print(f"Blend factor tensor shape after unsqueeze: {blend_factor_tensor.shape}")
        
        # Create explicit shape for expansion
        target_shape = list(s1.shape)
        print(f"Target shape for expansion: {target_shape}")
        print(f"Blend factor tensor shape before expansion: {blend_factor_tensor.shape}")
        
        # Apply blend mode with time-varying factors
        if blend_mode == "normal":
            # Expand blend_factor_tensor to match all dimensions
            blend_factor_expanded = blend_factor_tensor.expand(target_shape)
            print(f"Blend factor expanded shape: {blend_factor_expanded.shape}")
            print(f"Blend factor expanded stats - min: {blend_factor_expanded.min().item()}, max: {blend_factor_expanded.max().item()}, mean: {blend_factor_expanded.mean().item()}")
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
    "AudioLatentBlend": "Audio Latent Blend",
    "FlexlatentAudioBlend": "FlexlatentAudioBlend (BETA)",
} 