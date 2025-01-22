import numpy as np
import torch
from abc import abstractmethod
from ... import RyanOnTheInside
from ..flex.flex_base import FlexBase
from ...tooltips import apply_tooltips

@apply_tooltips
class FlexLatentBase(RyanOnTheInside, FlexBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "latents": ("LATENT",),
            },
            # Optional inputs are inherited from FlexBase
        }

    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/Latents"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_effect"

    def __init__(self):
        super().__init__()

    def process_below_threshold(self, latent, feature_value=None, **kwargs):
        """Default behavior for when feature value is below threshold: return latent unchanged."""
        return latent

    def apply_effect(
        self,
        latents,
        strength,
        feature_threshold,
        feature_param,
        feature_mode,
        opt_feature=None,
        **kwargs
    ):
        latents_np = latents["samples"].cpu().numpy()

        num_frames = latents_np.shape[0]

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            # Get the appropriate latent frame, handling possible shorter sequences
            latent = latents_np[i % latents_np.shape[0]]
            
            # Get feature value
            feature_value = self.get_feature_value(i, opt_feature)
            
            # Process parameters using FlexBase functionality
            processed_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=feature_value,
                feature_threshold=feature_threshold,
                strength=strength,
                feature_param=feature_param,
                feature_mode=feature_mode,
                **kwargs
            )

            # Ensure frame_index is included in processed_kwargs
            processed_kwargs['frame_index'] = i

            # Determine if effect should be applied based on feature value and threshold
            if feature_value is not None and feature_value >= processed_kwargs['feature_threshold']:
                processed_latent = self.apply_effect_internal(
                    latent,
                    **processed_kwargs
                )
            else:
                processed_latent = self.process_below_threshold(
                    latent,
                    **processed_kwargs
                )

            result.append(processed_latent)
            self.update_progress()

        self.end_progress()

        # Stack results and convert back to tensor
        result_np = np.stack(result)
        result_tensor = torch.from_numpy(result_np).float()

        return ({"samples": result_tensor},)

    @abstractmethod
    def apply_effect_internal(self, latent: np.ndarray, feature_value: float, strength: float, 
                            feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        """Apply the effect with processed parameters. To be implemented by child classes."""
        pass

