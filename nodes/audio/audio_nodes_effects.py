from .audio_nodes import AudioNodeBase
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F 
import librosa
import numpy as np

class AudioEffect(AudioNodeBase):
    def __init__(self):
        super().__init__()

    CATEGORY = "RyanOnTheInside/Audio/Effects"

class AudioPitchShift(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "n_steps": ("INT", {"default": 0, "min": -12, "max": 12, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pitch_shift_audio"

    def pitch_shift_audio(self, audio, n_steps):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        # Use the functional API for pitch shifting
        shifted_waveform = F.pitch_shift(waveform, sample_rate, n_steps=n_steps)
        return ({"waveform": shifted_waveform, "sample_rate": sample_rate},)


class AudioFade(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fade_in_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "fade_out_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "shape": (["linear", "exponential", "logarithmic", "quarter_sine", "half_sine"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "fade_audio"

    def fade_audio(self, audio, fade_in_duration, fade_out_duration, shape):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        fade_in_samples = int(fade_in_duration * sample_rate)
        fade_out_samples = int(fade_out_duration * sample_rate)
        fader = T.Fade(
            fade_in_len=fade_in_samples,
            fade_out_len=fade_out_samples,
            fade_shape=shape
        )
        faded_waveform = fader(waveform)
        return ({"waveform": faded_waveform, "sample_rate": sample_rate},)

class AudioGain(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_gain"

    def apply_gain(self, audio, gain_db):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        # Calculate the gain factor
        gain_factor = 10 ** (gain_db / 20)
        amplified_waveform = waveform * gain_factor
        return ({"waveform": amplified_waveform, "sample_rate": sample_rate},)

class AudioDither(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "bit_depth": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "noise_shaping": (["none", "triangular"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "dither_audio"

    def dither_audio(self, audio, bit_depth, noise_shaping):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        
        # Select density function based on noise shaping
        if noise_shaping == "triangular":
            density_function = "TPDF"
            noise_shaping_flag = True
        else:
            density_function = "RPDF"
            noise_shaping_flag = False

        # Apply dithering
        dithered_waveform = F.dither(
            waveform,
            density_function=density_function,
            noise_shaping=noise_shaping_flag
        )

        # Quantize the waveform to the specified bit depth
        max_val = 2 ** (bit_depth - 1) - 1
        quantized_waveform = torch.clamp(dithered_waveform, -1.0, 1.0)
        quantized_waveform = torch.round(quantized_waveform * max_val) / max_val

        return ({"waveform": quantized_waveform, "sample_rate": sample_rate},)
    
class AudioTimeStretch(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "rate": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "time_stretch"

    def time_stretch(self, audio, rate):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        
        # Ensure the input is 2D (batch, channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() > 3:
            raise ValueError("Input waveform has too many dimensions")
        
        # Convert to numpy for librosa processing
        waveform_np = waveform.squeeze(0).numpy()
        
        # Process each channel
        stretched_channels = []
        for channel in waveform_np:
            stretched = librosa.effects.time_stretch(channel, rate=rate)
            stretched_channels.append(stretched)
        
        # Stack channels and convert back to torch tensor
        stretched_waveform = torch.from_numpy(np.stack(stretched_channels)).unsqueeze(0)
        
        return ({"waveform": stretched_waveform, "sample_rate": sample_rate},)