import torch
import numpy as np
from abc import  abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
from ... import RyanOnTheInside
from ..flex.flex_base import FlexBase
from .audio_utils import pitch_shift, time_stretch
from ...tooltips import apply_tooltips

@apply_tooltips
class FlexAudioBase(FlexBase, RyanOnTheInside):
    @classmethod
    def INPUT_TYPES(cls):
        base_input_types = super().INPUT_TYPES()
        base_required = base_input_types.get("required", {})
        base_optional = base_input_types.get("optional", {})
        
        # Update required inputs
        base_required.update({
            "audio": ("AUDIO",),
            "target_fps": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 60.0, "step": 1.0}),
        })
        
        return {
            "required": base_required,
            "optional": base_optional,
        }

    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/Audio"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_effect"

    def __init__(self):
        super().__init__()


    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return []

    def apply_effect(self, audio, opt_feature=None, strength=1.0, feature_threshold=0.0, feature_param=None, feature_mode="relative", target_fps=3.0, **kwargs):
        waveform = audio['waveform']  # Shape: [Batch, Channels, Samples]
        sample_rate = audio['sample_rate']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        waveform = waveform.to(device)

        audio_duration = waveform.shape[-1] / sample_rate
        num_frames = int(audio_duration * target_fps)

        if opt_feature is not None:
            original_num_frames = opt_feature.frame_count
            num_frames = min(num_frames, original_num_frames)

        # Initialize parameter scheduler
        self.initialize_scheduler(num_frames, **kwargs)

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # Add batch dimension

        batch_size, num_channels, total_samples = waveform.shape
        samples_per_frame = max(1, total_samples // num_frames)

        # Initialize list to store processed frames
        processed_frames = []

        # Pre-compute averaged feature values if feature is provided
        if opt_feature is not None:
            feature_values = np.array([
                0.5 if self.get_feature_value(i, opt_feature) is None 
                else self.get_feature_value(i, opt_feature) 
                for i in range(opt_feature.frame_count)
            ])
            frames_per_segment = max(1, opt_feature.frame_count // num_frames)
            averaged_features = [np.mean(feature_values[i:i+frames_per_segment]) 
                                 for i in range(0, opt_feature.frame_count, frames_per_segment)]
        else:
            averaged_features = [None] * num_frames  # No feature modulation

        # Define cross-fade length (e.g., 10% of frame length)
        crossfade_length = int(samples_per_frame * 0.1)

        for i in range(num_frames):
            frame_start = i * samples_per_frame
            frame_end = min((i + 1) * samples_per_frame + crossfade_length, total_samples)
            audio_frame = waveform[:, :, frame_start:frame_end]  # Shape: [Batch, Channels, Samples]

            feature_value = averaged_features[i]
            
            # Process parameters using base class functionality
            processed_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=feature_value,
                feature_threshold=feature_threshold,
                strength=strength,
                feature_param=feature_param,
                feature_mode=feature_mode,
                sample_rate=sample_rate,
                **kwargs
            )

            if feature_value is not None and feature_value >= feature_threshold:
                try:
                    processed_frame = self.process_audio_frame(
                        audio_frame,feature_param=feature_param,
                        feature_mode=feature_mode,
                        **processed_kwargs
                    )
                except Exception as e:
                    import traceback
                    print(f"Error processing frame {i}:")
                    traceback.print_exc()
                    processed_frame = audio_frame
            else:
                # Process without feature modulation
                processed_frame = self.apply_effect_internal(audio_frame, **processed_kwargs)

            if i > 0:
                # Apply cross-fade with previous frame
                fade_in = torch.linspace(0, 1, crossfade_length).to(device)
                fade_out = 1 - fade_in
                crossfade_region = (processed_frames[-1][:, :, -crossfade_length:] * fade_out +
                                    processed_frame[:, :, :crossfade_length] * fade_in)
                processed_frames[-1][:, :, -crossfade_length:] = crossfade_region
                processed_frame = processed_frame[:, :, crossfade_length:]

            processed_frames.append(processed_frame)
            self.update_progress()

        self.end_progress()

        # Concatenate processed frames along the time dimension
        result_waveform = torch.cat(processed_frames, dim=-1).cpu()

        return ({"waveform": result_waveform, "sample_rate": sample_rate},)

    def process_audio_frame(self, audio_frame: torch.Tensor, feature_value: float, strength: float,
                            feature_param: str, feature_mode: str, **kwargs) -> torch.Tensor:
        # Modulate the selected parameter if feature_value is provided
        if feature_value is not None and feature_param in self.get_modifiable_params() and feature_param in kwargs:
            param_value = kwargs[feature_param]
            modulated_value = self.modulate_param(
                feature_param, param_value, feature_value, strength, feature_mode
            )
            # Ensure the modulated value is within the desired range
            kwargs[feature_param] = modulated_value

        # Call the child class's implementation
        return self.apply_effect_internal(audio_frame, **kwargs)

    @abstractmethod
    def apply_effect_internal(self, audio_frame: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply the effect to the audio frame. To be implemented by child classes."""
        pass

@apply_tooltips
class FlexAudioPitchShift(FlexAudioBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_input_types = super().INPUT_TYPES()
        base_required = base_input_types.get("required", {})
        base_optional = base_input_types.get("optional", {})
        base_required["feature_param"] = cls.get_modifiable_params()
        # Update required inputs
        base_required.update({
            "n_steps": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 12.0, "step": 0.1}),
        })



        return {
            "required": base_required,
            "optional": base_optional,
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["n_steps"]

    def apply_effect_internal(self, audio_frame: torch.Tensor, n_steps: float, **kwargs) -> torch.Tensor:
        sample_rate = kwargs.get('sample_rate', 44100)  # Default to 44100 if not provided

        # Use the value of n_steps for pitch shifting
        return pitch_shift(audio_frame, sample_rate, n_steps)

@apply_tooltips
class FlexAudioTimeStretch(FlexAudioBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_input_types = super().INPUT_TYPES()
        base_required = base_input_types.get("required", {})
        base_optional = base_input_types.get("optional", {})
        base_required["feature_param"] = cls.get_modifiable_params()
        # Update required inputs
        base_required.update({
            "rate": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
        })


        return {
            "required": base_required,
            "optional": base_optional,
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["rate"]

    def apply_effect_internal(self, audio_frame: torch.Tensor, rate: float, **kwargs) -> torch.Tensor:
        return time_stretch(audio_frame, rate)
