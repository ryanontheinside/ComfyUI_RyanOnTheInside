from .flex_latent_base import FlexLatentBase
import torch
import numpy as np
from scipy.signal import butter, sosfilt
from ..audio.audio_utils import (
    calculate_amplitude_envelope,
    calculate_rms_energy,
    calculate_spectral_flux,
    calculate_zero_crossing_rate
)
from ...tooltips import apply_tooltips

#NOTE just an experiment, it sucks
#TODO: get to this
@apply_tooltips
class LatentFrequencyBlender(FlexLatentBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "frequency_ranges": ("FREQUENCY_RANGE", {"multi": True}),
                "audio": ("AUDIO",),
                "feature_type": (
                    [
                        "amplitude_envelope",
                        "rms_energy",
                        "spectral_flux",
                        "zero_crossing_rate"
                    ],
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "feature_mode": (["relative", "absolute"], {"default": "relative"}),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "nonlinear_transform": (
                    ["none", "square", "sqrt", "log", "exp"],
                    {"default": "none"}
                ),
                "blending_mode": (["linear", "slerp", "hard_switch"], {"default": "linear"}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply_effect"
    CATEGORY = "RyanOnTheInside/ExperimentalWIP"

    @classmethod
    def get_modifiable_params(cls):
        return ["None"]

    def apply_effect(
        self,
        images,
        vae,
        frequency_ranges,
        audio,
        feature_type,
        strength,
        feature_mode,
        frame_rate,
        nonlinear_transform,
        blending_mode,
        **kwargs
    ):
        # Images come in as BHWC format
        # Ensure we only use RGB channels for VAE encoding
        images_for_vae = images[:, :, :, :3]
        encoded = vae.encode(images_for_vae)
        latents = {"samples": encoded}

        # Ensure frequency_ranges is a list
        if not isinstance(frequency_ranges, list):
            frequency_ranges = [frequency_ranges]

        if images.shape[0] != len(frequency_ranges):
            raise ValueError("Number of images must match number of frequency ranges.")

        # Calculate frame count from audio duration and frame rate
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        num_samples = waveform.shape[-1]
        audio_duration = num_samples / sample_rate
        frame_count = int(audio_duration * frame_rate)

        # Prepare latents
        latents_np = latents["samples"].cpu().numpy()
        num_frames = frame_count

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        # Initialize list to store feature values for each frequency range
        feature_values_per_range = []

        for freq_range in frequency_ranges:
            # Apply bandpass filter to get the frequency range
            filtered_audio = self._apply_bandpass_filter(audio, freq_range)

            # Calculate feature values based on selected feature_type
            if feature_type == "amplitude_envelope":
                feature_values = calculate_amplitude_envelope(
                    filtered_audio, frame_count, frame_rate
                )
            elif feature_type == "rms_energy":
                feature_values = calculate_rms_energy(
                    filtered_audio, frame_count, frame_rate
                )
            elif feature_type == "spectral_flux":
                feature_values = calculate_spectral_flux(
                    filtered_audio, frame_count, frame_rate
                )
            elif feature_type == "zero_crossing_rate":
                feature_values = calculate_zero_crossing_rate(
                    filtered_audio, frame_count, frame_rate
                )
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")

            # Apply nonlinear transformation to feature values
            feature_values = self._apply_nonlinear_transform(
                feature_values, nonlinear_transform
            )

            feature_values_per_range.append(feature_values)

        # Stack feature values and normalize
        feature_values_array = np.vstack(feature_values_per_range)
        # Avoid division by zero
        feature_values_array += 1e-8

        result = []
        for i in range(num_frames):
            # Get feature values for the current frame from each frequency range
            feature_values = feature_values_array[:, i]

            # Normalize feature values
            total = np.sum(feature_values)
            if total > 0:
                normalized_weights = feature_values / total
            else:
                # Equal weights if total is zero
                normalized_weights = np.ones(len(frequency_ranges)) / len(frequency_ranges)

            # Amplify weights using strength parameter
            normalized_weights *= strength

            # Normalize again after amplification
            total = np.sum(normalized_weights)
            if total > 0:
                normalized_weights /= total
            else:
                normalized_weights = np.ones(len(frequency_ranges)) / len(frequency_ranges)

            # Blend latents based on normalized weights
            blended_latent = None

            if blending_mode == "hard_switch":
                # Use the latent with the highest weight
                idx = np.argmax(normalized_weights)
                blended_latent = latents_np[idx % len(latents_np)]
            elif blending_mode == "slerp":
                # Initialize with first latent
                blended_latent = latents_np[0]
                # Iteratively apply spherical interpolation with weights
                for idx in range(1, len(latents_np)):
                    t = normalized_weights[idx]
                    blended_latent = self.spherical_interpolation(
                        blended_latent, 
                        latents_np[idx], 
                        t
                    )
            else:  # Linear blending
                for idx, weight in enumerate(normalized_weights):
                    latent_frame = latents_np[idx % len(latents_np)]
                    if blended_latent is None:
                        blended_latent = weight * latent_frame
                    else:
                        blended_latent += weight * latent_frame

            # Apply strength and feature mode
            if feature_mode == "relative":
                base_latent = latents_np[i % len(latents_np)]
                blended_latent = (1 - strength) * base_latent + strength * blended_latent
            else:  # Absolute
                blended_latent = strength * blended_latent

            result.append(blended_latent)
            self.update_progress()

        self.end_progress()

        result_np = np.stack(result)
        result_tensor = torch.from_numpy(result_np).float()

        return ({"samples": result_tensor},)

    def _apply_bandpass_filter(self, audio, freq_range):
        # Implement bandpass filter using scipy.signal
        order = freq_range.get('order', 4)
        low_cutoff = freq_range['low_cutoff']
        high_cutoff = freq_range['high_cutoff']
        sample_rate = audio['sample_rate']
        waveform = audio['waveform']

        # Ensure waveform is a NumPy array for processing
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        sos = butter(order, [low_cutoff, high_cutoff], btype='bandpass', fs=sample_rate, output='sos')
        filtered = sosfilt(sos, waveform)

        # Convert filtered waveform back to PyTorch tensor if needed
        return {'waveform': filtered, 'sample_rate': sample_rate}

    def _apply_nonlinear_transform(self, feature_values, transform_type):
        if transform_type == "square":
            return np.square(feature_values)
        elif transform_type == "sqrt":
            return np.sqrt(feature_values)
        elif transform_type == "log":
            return np.log1p(feature_values)
        elif transform_type == "exp":
            return np.exp(feature_values)
        else:  # "none"
            return feature_values

    def process_latent(self, latent: np.ndarray, **kwargs) -> np.ndarray:
        # This method is not used in this implementation since processing is done in apply_effect
        return latent
    
    #TODO contrast with base class version
    def spherical_interpolation(self, latent1, latent2, t):
        # Flatten the latents
        latent1_flat = latent1.flatten()
        latent2_flat = latent2.flatten()

        # Normalize the vectors
        norm1 = np.linalg.norm(latent1_flat) + 1e-8  # Avoid division by zero
        norm2 = np.linalg.norm(latent2_flat) + 1e-8
        latent1_norm = latent1_flat / norm1
        latent2_norm = latent2_flat / norm2

        # Compute dot product and omega
        dot = np.dot(latent1_norm, latent2_norm)
        dot = np.clip(dot, -1.0, 1.0)  # Ensure the dot product is within [-1, 1]
        omega = np.arccos(dot)

        # Check if omega is close to zero
        if np.isclose(omega, 0):
            # Use linear interpolation
            result_flat = (1 - t) * latent1_flat + t * latent2_flat
        else:
            sin_omega = np.sin(omega)
            coef1 = np.sin((1 - t) * omega) / sin_omega
            coef2 = np.sin(t * omega) / sin_omega
            # Interpolate using normalized vectors
            result_norm = coef1 * latent1_norm + coef2 * latent2_norm
            # Scale back to original magnitude (optional)
            interpolated_norm = (1 - t) * norm1 + t * norm2
            result_flat = result_norm * interpolated_norm

        # Reshape result to original latent shape
        result = result_flat.reshape(latent1.shape)
        return result