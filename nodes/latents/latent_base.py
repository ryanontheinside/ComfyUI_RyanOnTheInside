import cv2
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
from ... import RyanOnTheInside

#NOTE work in progress, not even close to being finished
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
        latents_np = latents["samples"].cpu().numpy()  # Access the "samples" key

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            latent = latents_np[i]
            feature_value = feature.get_value_at_frame(i)
            kwargs['frame_index'] = i

            if feature_value >= feature_threshold:
                processed_latent = self.process_latent(latent, feature_value, strength,
                                                       feature_param=feature_param,
                                                       feature_mode=feature_mode,
                                                       **kwargs)
            else:
                processed_latent = latent

            result.append(processed_latent)
            self.update_progress()

        self.end_progress()

        # Convert the list of numpy arrays to a single numpy array
        result_np = np.stack(result)

        # Convert the numpy array to a PyTorch tensor (already in BCHW format)
        result_tensor = torch.from_numpy(result_np).float()

        return ({"samples": result_tensor},)  # Return as a dictionary with "samples" key

    def process_latent(self, latent: np.ndarray, feature_value: float, strength: float,
                       feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                if param_name == feature_param:
                    kwargs[param_name] = self.modulate_param(param_name, kwargs[param_name],
                                                             feature_value, strength, feature_mode)

        # Call the child class's implementation
        return self.apply_effect_internal(latent, **kwargs)

    @abstractmethod
    def apply_effect_internal(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the latent. To be implemented by child classes."""

#NOTE work in progress
class FlexLatentInterpolate(FlexLatentBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "latent_2": ("LATENT",),
                "interpolation_mode": (["Linear", "Spherical"], {"default": "Linear"}),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["interpolation_mode", "None"]

    def apply_effect(self, latents, feature, feature_pipe, strength, feature_threshold, feature_param, feature_mode, latent_2, interpolation_mode, **kwargs):
        num_frames = feature_pipe.frame_count
        latents_np = latents["samples"].cpu().numpy()
        latent_2_np = latent_2["samples"].cpu().numpy()

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            latent = latents_np[i]
            latent_2_frame = latent_2_np[i]
            feature_value = feature.get_value_at_frame(i)
            kwargs['frame_index'] = i

            if feature_value >= feature_threshold:
                processed_latent = self.process_latent(latent, feature_value, strength,
                                                       feature_param=feature_param,
                                                       feature_mode=feature_mode,
                                                       latent_2=latent_2_frame,
                                                       interpolation_mode=interpolation_mode,
                                                       **kwargs)
            else:
                processed_latent = latent

            result.append(processed_latent)
            self.update_progress()

        self.end_progress()

        result_np = np.stack(result)
        result_tensor = torch.from_numpy(result_np).float()

        return ({"samples": result_tensor},)

    def process_latent(self, latent: np.ndarray, feature_value: float, strength: float,
                       feature_param: str, feature_mode: str, latent_2: np.ndarray,
                       interpolation_mode: str, **kwargs) -> np.ndarray:
        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                if param_name == feature_param:
                    kwargs[param_name] = self.modulate_param(param_name, kwargs[param_name],
                                                             feature_value, strength, feature_mode)

        # Call the apply_effect_internal method
        return self.apply_effect_internal(latent, latent_2, interpolation_mode, feature_value, strength)

    def apply_effect_internal(self, latent: np.ndarray, latent_2: np.ndarray, interpolation_mode: str, feature_value: float, strength: float) -> np.ndarray:
        t = feature_value * strength
        t = np.clip(t, 0.0, 1.0)  # Ensure interpolation factor is within [0,1]

        if interpolation_mode == "Linear":
            # Simple linear interpolation
            result = (1 - t) * latent + t * latent_2
        else:  # Spherical interpolation
            # Flatten the latents
            latent_flat = latent.reshape(-1)
            latent_2_flat = latent_2.reshape(-1)

            # Compute dot product and norms
            dot = np.dot(latent_flat, latent_2_flat)
            norms = np.linalg.norm(latent_flat) * np.linalg.norm(latent_2_flat)
            omega = np.arccos(np.clip(dot / norms, -1.0, 1.0))

            if omega == 0:
                result = latent
            else:
                sin_omega = np.sin(omega)
                coef1 = np.sin((1 - t) * omega) / sin_omega
                coef2 = np.sin(t * omega) / sin_omega
                result = coef1 * latent + coef2 * latent_2

        return result
    


class EmbeddingGuidedLatentInterpolate(FlexLatentBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "latent_2": ("LATENT",),
                "embedding_1": ("EMBEDS",),  # New input for the first embedding
                "embedding_2": ("EMBEDS",),  # New input for the second embedding
                "interpolation_mode": (["Linear", "Spherical"], {"default": "Linear"}),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["interpolation_mode", "None"]

    def apply_effect(self, latents, feature, feature_pipe, strength, feature_threshold, feature_param, feature_mode,
                     latent_2, embedding_1, embedding_2, interpolation_mode, **kwargs):
        num_frames = feature_pipe.frame_count
        latents_np = latents["samples"].cpu().numpy()
        latent_2_np = latent_2["samples"].cpu().numpy()
        embedding_1_np = embedding_1.cpu().numpy()
        embedding_2_np = embedding_2.cpu().numpy()

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            latent = latents_np[i]
            latent_2_frame = latent_2_np[i]
            emb1 = embedding_1_np[i]
            emb2 = embedding_2_np[i]
            feature_value = feature.get_value_at_frame(i)
            kwargs['frame_index'] = i

            if feature_value >= feature_threshold:
                processed_latent = self.process_latent(
                    latent,
                    feature_value,
                    strength,
                    feature_param=feature_param,
                    feature_mode=feature_mode,
                    latent_2=latent_2_frame,
                    embedding_1=emb1,
                    embedding_2=emb2,
                    interpolation_mode=interpolation_mode,
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

    def process_latent(self, latent: np.ndarray, feature_value: float, strength: float,
                       feature_param: str, feature_mode: str, latent_2: np.ndarray,
                       embedding_1: np.ndarray, embedding_2: np.ndarray,
                       interpolation_mode: str, **kwargs) -> np.ndarray:
        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                if param_name == feature_param:
                    kwargs[param_name] = self.modulate_param(param_name, kwargs[param_name],
                                                             feature_value, strength, feature_mode)

        # Call the apply_effect_internal method with embeddings
        return self.apply_effect_internal(
            latent,
            latent_2,
            embedding_1,
            embedding_2,
            interpolation_mode,
            feature_value,
            strength
        )

    def apply_effect_internal(self, latent: np.ndarray, latent_2: np.ndarray,
                              embedding_1: np.ndarray, embedding_2: np.ndarray,
                              interpolation_mode: str, feature_value: float, strength: float) -> np.ndarray:
        t = feature_value * strength
        t = np.clip(t, 0.0, 1.0)  # Ensure interpolation factor is within [0,1]

        # Normalize embeddings
        emb1_norm = embedding_1 / (np.linalg.norm(embedding_1) + 1e-8)
        emb2_norm = embedding_2 / (np.linalg.norm(embedding_2) + 1e-8)

        # Compute cosine similarity between embeddings
        similarity = np.dot(emb1_norm.flatten(), emb2_norm.flatten())

        # Adjust interpolation factor based on similarity
        # Here, we can define how similarity influences 't'
        # For example, increase 't' if similarity is high
        similarity_factor = (similarity + 1) / 2  # Normalize similarity to [0, 1]
        t_adjusted = t * similarity_factor

        if interpolation_mode == "Linear":
            # Linear interpolation with adjusted 't'
            result = (1 - t_adjusted) * latent + t_adjusted * latent_2
        else:  # Spherical interpolation
            # Flatten the latents
            latent_flat = latent.reshape(-1)
            latent_2_flat = latent_2.reshape(-1)

            # Compute dot product and norms
            dot_product = np.dot(latent_flat, latent_2_flat)
            norms = (np.linalg.norm(latent_flat) * np.linalg.norm(latent_2_flat)) + 1e-8
            omega = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))

            if omega == 0:
                result = latent
            else:
                sin_omega = np.sin(omega)
                coef1 = np.sin((1 - t_adjusted) * omega) / sin_omega
                coef2 = np.sin(t_adjusted * omega) / sin_omega
                result = coef1 * latent + coef2 * latent_2

            # Reshape result back to original latent shape
            result = result.reshape(latent.shape)

        return result
