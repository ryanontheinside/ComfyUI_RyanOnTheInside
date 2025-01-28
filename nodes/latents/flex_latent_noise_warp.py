from .flex_latent_base import FlexLatentBase    
from .noise_warp import NoiseWarper
from ... import apply_tooltips
import numpy as np
import torch
@apply_tooltips
class FlexWarpedNoise(FlexLatentBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = (["dx", "dy", "post_noise_alpha", "progressive_noise_alpha"],)
        base_inputs["required"].update({
            "scale_factor": ("INT", {"default": 1, "min": 1, "max": 8}),
            "dx": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
            "dy": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0}),
            "post_noise_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            "progressive_noise_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            "latent_shape": (["BTCHW", "BCTHW", "BCHW"], {"default": "BCHW"}),
        })
        base_inputs["optional"].update({
            "model": ("MODEL",),
            "sigmas": ("SIGMAS",),
        })
        return base_inputs

    RETURN_TYPES = ("LATENT", "IMAGE", "IMAGE")
    RETURN_NAMES = ("noise", "visualization", "optical_flows")
   
    @classmethod
    def get_modifiable_params(cls):
        return ["dx", "dy", "post_noise_alpha", "progressive_noise_alpha"]

    def __init__(self):
        super().__init__()
        self.noise_warper = None

    def apply_effect_internal(self, latent: np.ndarray, feature_value: float, strength: float,
                            feature_param: str, feature_mode: str, scale_factor: int = 1,
                            dx: float = 0.0, dy: float = 0.0, post_noise_alpha: float = 0.0,
                            progressive_noise_alpha: float = 0.0, frame_index: int = 0, **kwargs) -> np.ndarray:
        
        # After FlexLatentBase removes B, we get the full temporal sequence for this batch item
        if len(latent.shape) == 4:  # We have a sequence (TCHW or CTHW)
            if latent.shape[0] > latent.shape[1]:  # TCHW format
                frames, channels, height, width = latent.shape
                warped_sequence = np.zeros_like(latent)
                for t in range(frames):
                    warped_sequence[t] = self._warp_single_frame(
                        latent[t], channels, height, width,
                        scale_factor, dx, dy, post_noise_alpha, progressive_noise_alpha
                    )
                return warped_sequence
            else:  # CTHW format
                channels, frames, height, width = latent.shape
                warped_sequence = np.zeros_like(latent)
                for t in range(frames):
                    warped_sequence[:, t] = self._warp_single_frame(
                        latent[:, t], channels, height, width,
                        scale_factor, dx, dy, post_noise_alpha, progressive_noise_alpha
                    )
                return warped_sequence
        else:  # CHW format - single frame
            channels, height, width = latent.shape
            return self._warp_single_frame(
                latent, channels, height, width,
                scale_factor, dx, dy, post_noise_alpha, progressive_noise_alpha
            )

    def _warp_single_frame(self, frame, channels, height, width, scale_factor, dx, dy, post_noise_alpha, progressive_noise_alpha):
        # Convert frame to tensor and move to device
        frame_tensor = torch.from_numpy(frame).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create or recreate noise warper with this frame's noise
        self.noise_warper = NoiseWarper(
            c=channels, 
            h=height, 
            w=width,
            device="cuda" if torch.cuda.is_available() else "cpu",
            scale_factor=scale_factor,
            post_noise_alpha=post_noise_alpha,
            progressive_noise_alpha=progressive_noise_alpha,
        )

        # Create displacement maps using the already-modulated parameters
        dx_map = np.full((height, width), dx)
        dy_map = np.full((height, width), dy)

        # Convert to torch tensors
        dx_tensor = torch.from_numpy(dx_map).to(self.noise_warper._state.device)
        dy_tensor = torch.from_numpy(dy_map).to(self.noise_warper._state.device)

        # Apply warping
        self.noise_warper(dx_tensor, dy_tensor)
        return self.noise_warper.noise.cpu().numpy()
