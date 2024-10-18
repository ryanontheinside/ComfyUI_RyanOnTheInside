# custom_nodes/ComfyUI_RyanOnTheInside/nodes/latents/latent_base.py

import numpy as np
import torch
from abc import ABC, abstractmethod
from comfy.utils import ProgressBar
from ... import RyanOnTheInside

class FlexLatentBase(RyanOnTheInside, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
                "feature": ("FEATURE",),
                "feature_pipe": ("FEATURE_PIPE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_param": (cls.get_modifiable_params(), {"default": cls.get_modifiable_params()[0]}),
                "feature_mode": (["relative", "absolute"], {"default": "relative"}),
            }
        }

    CATEGORY = "RyanOnTheInside/FlexLatent"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_effect"

    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return []

    def modulate_param(self, param_name, param_value, feature_value, strength, mode):
        if mode == "relative":
            return param_value * (1 + (feature_value - 0.5) * strength)
        else:  # absolute
            return param_value * feature_value * strength

    def apply_effect(self, latents, feature, feature_pipe, strength, feature_threshold, feature_param, feature_mode, **kwargs):
        num_frames = feature_pipe.frame_count
        latents_np = latents["samples"].cpu().numpy()

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            latent = latents_np[i]
            feature_value = feature.get_value_at_frame(i)
            kwargs['frame_index'] = i
            kwargs['feature_value'] = feature_value
            kwargs['strength'] = strength
            kwargs['feature_param'] = feature_param
            kwargs['feature_mode'] = feature_mode

            if feature_value >= feature_threshold:
                processed_latent = self.process_latent(latent, **kwargs)
            else:
                processed_latent = latent

            result.append(processed_latent)
            self.update_progress()

        self.end_progress()

        result_np = np.stack(result)
        result_tensor = torch.from_numpy(result_np).float()

        return ({"samples": result_tensor},)

    @abstractmethod
    def process_latent(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        """Process the latent using subclass-specific logic."""

# custom_nodes/ComfyUI_RyanOnTheInside/nodes/latents/flex_latent_interpolate.py

import numpy as np

class FlexLatentInterpolate(FlexLatentBase):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "latent_2": ("LATENT",),
            "interpolation_mode": (["Linear", "Spherical"], {"default": "Linear"}),
        })
        return inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["interpolation_mode", "None"]

    def process_latent(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        feature_value = kwargs['feature_value']
        strength = kwargs['strength']
        feature_param = kwargs['feature_param']
        feature_mode = kwargs['feature_mode']
        latent_2 = kwargs['latent_2']
        interpolation_mode = kwargs['interpolation_mode']

        # Modulate any parameters if needed
        if feature_param in self.get_modifiable_params():
            modulated_value = self.modulate_param(
                feature_param,
                kwargs.get(feature_param, 0),
                feature_value,
                strength,
                feature_mode
            )
            kwargs[feature_param] = modulated_value

        latent_2_np = latent_2["samples"].cpu().numpy()[kwargs['frame_index']]

        # Perform interpolation
        t = np.clip(feature_value * strength, 0.0, 1.0)
        if interpolation_mode == "Linear":
            result = (1 - t) * latent + t * latent_2_np
        else:  # Spherical interpolation
            result = self.spherical_interpolation(latent, latent_2_np, t)
        return result

    def spherical_interpolation(self, latent1, latent2, t):
        # Flatten the latents
        latent1_flat = latent1.flatten()
        latent2_flat = latent2.flatten()

        # Compute dot product and norms
        dot = np.dot(latent1_flat, latent2_flat)
        norm1 = np.linalg.norm(latent1_flat)
        norm2 = np.linalg.norm(latent2_flat)
        denominator = norm1 * norm2 + 1e-8  # Avoid division by zero
        omega = np.arccos(np.clip(dot / denominator, -1.0, 1.0))

        if np.isclose(omega, 0):
            return latent1
        else:
            sin_omega = np.sin(omega)
            coef1 = np.sin((1 - t) * omega) / sin_omega
            coef2 = np.sin(t * omega) / sin_omega
            result = coef1 * latent1 + coef2 * latent2
            return result.reshape(latent1.shape)
        
# custom_nodes/ComfyUI_RyanOnTheInside/nodes/latents/embedding_guided_latent_interpolate.py

    def spherical_interpolation(self, latent1, latent2, t):
        # Flatten the latents
        latent1_flat = latent1.flatten()
        latent2_flat = latent2.flatten()

        # Compute dot product and norms
        dot = np.dot(latent1_flat, latent2_flat)
        norm1 = np.linalg.norm(latent1_flat)
        norm2 = np.linalg.norm(latent2_flat)
        denominator = norm1 * norm2 + 1e-8  # Avoid division by zero
        omega = np.arccos(np.clip(dot / denominator, -1.0, 1.0))

        if np.isclose(omega, 0):
            return latent1
        else:
            sin_omega = np.sin(omega)
            coef1 = np.sin((1 - t) * omega) / sin_omega
            coef2 = np.sin(t * omega) / sin_omega
            result = coef1 * latent1 + coef2 * latent2
            return result.reshape(latent1.shape)

import numpy as np

class EmbeddingGuidedLatentInterpolate(FlexLatentBase):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "latent_2": ("LATENT",),
            "embedding_1": ("EMBEDS",),
            "embedding_2": ("EMBEDS",),
            "interpolation_mode": (["Linear", "Spherical"], {"default": "Linear"}),
        })
        return inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["interpolation_mode", "None"]

    def process_latent(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        feature_value = kwargs['feature_value']
        strength = kwargs['strength']
        feature_param = kwargs['feature_param']
        feature_mode = kwargs['feature_mode']
        latent_2 = kwargs['latent_2']
        embedding_1 = kwargs['embedding_1']
        embedding_2 = kwargs['embedding_2']
        interpolation_mode = kwargs['interpolation_mode']
        frame_index = kwargs['frame_index']

        # Modulate any parameters if needed
        if feature_param in self.get_modifiable_params():
            modulated_value = self.modulate_param(
                feature_param,
                kwargs.get(feature_param, 0),
                feature_value,
                strength,
                feature_mode
            )
            kwargs[feature_param] = modulated_value

        latent_2_np = latent_2["samples"].cpu().numpy()[frame_index]
        embedding_1_np = embedding_1.cpu().numpy()[frame_index]
        embedding_2_np = embedding_2.cpu().numpy()[frame_index]

        # Compute similarity between embeddings
        similarity = self.compute_similarity(embedding_1_np, embedding_2_np)

        # Adjust interpolation factor based on similarity
        t = np.clip(feature_value * strength * similarity, 0.0, 1.0)

        if interpolation_mode == "Linear":
            result = (1 - t) * latent + t * latent_2_np
        else:  # Spherical interpolation
            result = self.spherical_interpolation(latent, latent_2_np, t)
        return result

    def compute_similarity(self, emb1, emb2):
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        similarity = np.dot(emb1_norm.flatten(), emb2_norm.flatten())
        # Normalize similarity to [0, 1]
        similarity = (similarity + 1) / 2
        return similarity

    def spherical_interpolation(self, latent1, latent2, t):
        # Same as in the previous subclass
        latent1_flat = latent1.flatten()
        latent2_flat = latent2.flatten()

        dot = np.dot(latent1_flat, latent2_flat)
        norm1 = np.linalg.norm(latent1_flat)
        norm2 = np.linalg.norm(latent2_flat)
        denominator = norm1 * norm2 + 1e-8
        omega = np.arccos(np.clip(dot / denominator, -1.0, 1.0))

        if np.isclose(omega, 0):
            return latent1
        else:
            sin_omega = np.sin(omega)
            coef1 = np.sin((1 - t) * omega) / sin_omega
            coef2 = np.sin(t * omega) / sin_omega
            result = coef1 * latent1 + coef2 * latent2
            return result.reshape(latent1.shape)
        

class FlexLatentBlend(FlexLatentBase):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "latent_2": ("LATENT",),
            "blend_mode": (["Add", "Multiply", "Screen", "Overlay"], {"default": "Add"}),
            "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["blend_strength", "None"]

    def process_latent(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        feature_value = kwargs['feature_value']
        strength = kwargs['strength']
        feature_param = kwargs['feature_param']
        feature_mode = kwargs['feature_mode']
        latent_2 = kwargs['latent_2']
        blend_mode = kwargs['blend_mode']
        blend_strength = kwargs['blend_strength']
        frame_index = kwargs['frame_index']

        # Modulate the blend_strength parameter if selected
        if feature_param == "blend_strength":
            blend_strength = self.modulate_param(
                "blend_strength",
                blend_strength,
                feature_value,
                strength,
                feature_mode
            )
            # Ensure blend_strength remains within [0, 1]
            blend_strength = np.clip(blend_strength, 0.0, 1.0)

        latent_2_np = latent_2["samples"].cpu().numpy()[frame_index]

        # Apply blending operation
        blended_latent = self.apply_blend(latent, latent_2_np, blend_mode)
        # Interpolate between original and blended latent based on blend_strength
        result = (1 - blend_strength) * latent + blend_strength * blended_latent

        return result

    def apply_blend(self, latent1, latent2, mode):
        if mode == "Add":
            return latent1 + latent2
        elif mode == "Multiply":
            return latent1 * latent2
        elif mode == "Screen":
            return 1 - (1 - latent1) * (1 - latent2)
        elif mode == "Overlay":
            return np.where(latent1 < 0.5,
                            2 * latent1 * latent2,
                            1 - 2 * (1 - latent1) * (1 - latent2))
        else:
            # Default to Add if mode is unrecognized
            return latent1 + latent2
        
class FlexLatentNoise(FlexLatentBase):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "noise_level": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "noise_type": (["Gaussian", "Uniform"], {"default": "Gaussian"}),
        })
        return inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["noise_level", "None"]

    def process_latent(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        feature_value = kwargs['feature_value']
        strength = kwargs['strength']
        feature_param = kwargs['feature_param']
        feature_mode = kwargs['feature_mode']
        noise_level = kwargs['noise_level']
        noise_type = kwargs['noise_type']

        # Modulate the noise_level parameter if selected
        if feature_param == "noise_level":
            noise_level = self.modulate_param(
                "noise_level",
                noise_level,
                feature_value,
                strength,
                feature_mode
            )
            # Ensure noise_level remains within [0, 1]
            noise_level = np.clip(noise_level, 0.0, 1.0)

        # Generate noise
        if noise_type == "Gaussian":
            noise = np.random.randn(*latent.shape) * noise_level
        elif noise_type == "Uniform":
            noise = (np.random.rand(*latent.shape) - 0.5) * 2 * noise_level
        else:
            noise = np.zeros_like(latent)

        # Add noise to the latent
        result = latent + noise

        return result