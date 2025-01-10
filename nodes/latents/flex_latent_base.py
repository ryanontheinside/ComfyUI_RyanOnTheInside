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

        # Determine frame count from either feature, latents, or longest parameter list
        if opt_feature is not None:
            num_frames = opt_feature.frame_count
        else:
            # Start with number of input frames
            num_frames = latents_np.shape[0]
            # Check all parameters for lists/arrays that might be longer
            for value in kwargs.values():
                if isinstance(value, (list, tuple, np.ndarray)):
                    num_frames = max(num_frames, len(value))

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            # Get the appropriate latent frame, handling possible shorter sequences
            latent = latents_np[i % latents_np.shape[0]]
            
            # Set frame index for parameter processing
            kwargs['frame_index'] = i

            # Get feature value (0.5 if no feature provided)
            feature_value = self.get_feature_value(i, opt_feature)
            feature_value = 0.5 if feature_value is None else feature_value

            # Process parameters based on feature value
            processed_kwargs = {}
            for param_name in self.get_modifiable_params():
                if param_name in kwargs:
                    param_value = kwargs[param_name]
                    if isinstance(param_value, (list, tuple, np.ndarray)):
                        try:
                            base_value = float(param_value[i])
                        except (IndexError, TypeError):
                            base_value = float(param_value[0])
                    else:
                        base_value = float(param_value)
                    
                    processed_kwargs[param_name] = self.modulate_param(
                        param_name, base_value, feature_value, strength, feature_mode
                    )

            # Add remaining kwargs
            for key, value in kwargs.items():
                if key not in processed_kwargs and key != 'frame_index':
                    if isinstance(value, (list, tuple, np.ndarray)):
                        try:
                            processed_kwargs[key] = value[i]
                        except (IndexError, TypeError):
                            processed_kwargs[key] = value[0]
                    else:
                        processed_kwargs[key] = value

            # Add feature-related values to kwargs
            processed_kwargs.update({
                'feature_value': feature_value,
                'strength': strength,
                'feature_param': feature_param,
                'feature_mode': feature_mode,
                'frame_index': i
            })

            # Process the latent
            if opt_feature is None or feature_value >= feature_threshold:
                processed_latent = self.apply_effect_internal(latent, **processed_kwargs)
            else:
                processed_latent = latent

            result.append(processed_latent)
            self.update_progress()

        self.end_progress()

        # Stack results and convert back to tensor
        result_np = np.stack(result)
        result_tensor = torch.from_numpy(result_np).float()

        return ({"samples": result_tensor},)

    @abstractmethod
    def apply_effect_internal(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        """To be implemented by subclasses."""
        pass

