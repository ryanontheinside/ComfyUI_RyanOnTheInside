import numpy as np
import torch
from abc import abstractmethod
from ... import RyanOnTheInside
from ..flex.flex_base import FlexBase

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

    CATEGORY = "RyanOnTheInside/FlexLatent"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_effect"

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
            latent = latents_np[i]
            feature_value = 1.0  # Default feature value
            apply_effect = True  # Default to applying the effect

            if opt_feature is not None:
                feature_value = opt_feature.get_value_at_frame(i)
                apply_effect = feature_value >= feature_threshold

            kwargs['frame_index'] = i
            kwargs['feature_value'] = feature_value  # Add feature_value to kwargs
            kwargs['strength'] = strength            # Add strength to kwargs
            kwargs['feature_param'] = feature_param  # Add feature_param to kwargs
            kwargs['feature_mode'] = feature_mode    # Add feature_mode to kwargs

            if apply_effect:
                processed_latent = self.process_latent(
                    latent,
                    **kwargs
                )
            else:
                processed_latent = latent

            result.append(processed_latent)
            self.update_progress()

        self.end_progress()

        result_np = np.stack(result)
        result_tensor = torch.from_numpy(result_np).float()

        return ({"samples": result_tensor},)

    def process_latent(
        self,
        latent: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        feature_value = kwargs['feature_value']
        strength = kwargs['strength']
        feature_param = kwargs['feature_param']
        feature_mode = kwargs['feature_mode']

        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs and param_name == feature_param:
                kwargs[param_name] = self.modulate_param(
                    param_name,
                    kwargs[param_name],
                    feature_value,
                    strength,
                    feature_mode
                )

        # Call the subclass's implementation
        return self.apply_effect_internal(latent, **kwargs)

    @abstractmethod
    def apply_effect_internal(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        """To be implemented by subclasses."""
        pass

