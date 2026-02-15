import numpy as np
import torch
from abc import abstractmethod
from ... import RyanOnTheInside
from ..flex.flex_base import FlexBase
from ...tooltips import apply_tooltips

def _is_audio_latent(samples_np):
    """Detect ACE-Step 1.5 audio latents: shape [B, 64, T] with B typically 1."""
    return samples_np.ndim == 3 and samples_np.shape[1] == 64

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

    def _map_frame_index(self, time_step, num_time_steps, feature):
        """Map an audio time step index to the corresponding feature frame index."""
        if feature is None:
            return time_step
        feature_frames = feature.frame_count
        return int(time_step * feature_frames / num_time_steps)

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

        if _is_audio_latent(latents_np):
            result_tensor = self._apply_effect_audio(
                latents_np, strength, feature_threshold, feature_param,
                feature_mode, opt_feature, **kwargs
            )
        else:
            result_tensor = self._apply_effect_spatial(
                latents_np, strength, feature_threshold, feature_param,
                feature_mode, opt_feature, **kwargs
            )

        return ({"samples": result_tensor},)

    def _apply_effect_spatial(
        self, latents_np, strength, feature_threshold, feature_param,
        feature_mode, opt_feature, **kwargs
    ):
        """Original per-batch-frame iteration for image/video latents."""
        num_frames = latents_np.shape[0]
        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            latent = latents_np[i % latents_np.shape[0]]
            feature_value = self.get_feature_value(i, opt_feature)

            processed_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=feature_value,
                feature_threshold=feature_threshold,
                strength=strength,
                feature_param=feature_param,
                feature_mode=feature_mode,
                **kwargs
            )
            processed_kwargs['frame_index'] = i

            # Default feature_value to 1.0 when no feature is connected
            effective_value = feature_value if feature_value is not None else 1.0
            processed_kwargs['feature_value'] = effective_value
            if effective_value >= processed_kwargs['feature_threshold']:
                processed_latent = self.apply_effect_internal(latent, **processed_kwargs)
            else:
                processed_latent = self.process_below_threshold(latent, **processed_kwargs)

            result.append(processed_latent)
            self.update_progress()

        self.end_progress()
        result_np = np.stack(result)
        return torch.from_numpy(result_np).float()

    def _apply_effect_audio(
        self, latents_np, strength, feature_threshold, feature_param,
        feature_mode, opt_feature, **kwargs
    ):
        """Per-time-step iteration for 1D audio latents [B, C, T]."""
        num_time_steps = latents_np.shape[-1]  # T dimension
        self.start_progress(num_time_steps, desc=f"Applying {self.__class__.__name__} (audio)")

        # Work on first batch item (audio is typically B=1)
        # latents_np shape: [B, C, T]
        result = latents_np.copy()

        for t in range(num_time_steps):
            # Map audio time step to feature frame index
            feature_idx = self._map_frame_index(t, num_time_steps, opt_feature)
            feature_value = self.get_feature_value(feature_idx, opt_feature)

            processed_kwargs = self.process_parameters(
                frame_index=feature_idx,
                feature_value=feature_value,
                feature_threshold=feature_threshold,
                strength=strength,
                feature_param=feature_param,
                feature_mode=feature_mode,
                **kwargs
            )
            processed_kwargs['frame_index'] = t
            processed_kwargs['_audio_mode'] = True

            # Default feature_value to 1.0 when no feature is connected
            effective_value = feature_value if feature_value is not None else 1.0
            processed_kwargs['feature_value'] = effective_value

            # Extract column [C] for each batch item, apply effect, write back
            for b in range(latents_np.shape[0]):
                column = latents_np[b, :, t]  # shape [C]

                if effective_value >= processed_kwargs['feature_threshold']:
                    processed_column = self.apply_effect_internal(column, **processed_kwargs)
                else:
                    processed_column = self.process_below_threshold(column, **processed_kwargs)

                result[b, :, t] = processed_column

            self.update_progress()

        self.end_progress()
        return torch.from_numpy(result).float()

    @abstractmethod
    def apply_effect_internal(self, latent: np.ndarray, feature_value: float, strength: float,
                            feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        """Apply the effect with processed parameters. To be implemented by child classes.

        For spatial latents: latent is [C, H, W] (one frame).
        For audio latents: latent is [C] (one time step column).
        """
        pass
