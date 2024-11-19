from .audio_nodes import AudioNodeBase
from .audio_utils import (
    pad_audio,
    normalize_volume,
    resample_audio,
    merge_channels,
    split_channels,
    concatenate_audio,
    combine_audio,
    dither_audio,
)
import torch

class AudioUtility(AudioNodeBase):
    def __init__(self):
        super().__init__()

    CATEGORY = "RyanOnTheInside/Audio/Utility"

class AudioPad(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 44100, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 44100, "step": 1}),
                "pad_mode": (["constant", "reflect", "replicate", "circular"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pad_audio_node"

    def pad_audio_node(self, audio, pad_left, pad_right, pad_mode):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        padded_waveform = pad_audio(waveform, pad_left, pad_right, pad_mode)
        return ({"waveform": padded_waveform, "sample_rate": sample_rate},)

class AudioVolumeNormalization(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_level": ("FLOAT", {"default": -10.0, "min": -60.0, "max": 0.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "normalize_volume_node"

    def normalize_volume_node(self, audio, target_level):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        normalized_waveform = normalize_volume(waveform, target_level)
        return ({"waveform": normalized_waveform, "sample_rate": sample_rate},)

class AudioResample(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "new_sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 100}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "resample_audio_node"

    def resample_audio_node(self, audio, new_sample_rate):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        resampled_waveform = resample_audio(waveform, sample_rate, new_sample_rate)
        return ({"waveform": resampled_waveform, "sample_rate": new_sample_rate},)

class AudioChannelMerge(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("LIST[AUDIO]",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "merge_channels_node"

    def merge_channels_node(self, audio_list):
        waveform_list = [audio['waveform'] for audio in audio_list]
        sample_rate = audio_list[0]['sample_rate']

        # Check that all sample rates are the same
        for audio in audio_list:
            if audio['sample_rate'] != sample_rate:
                raise ValueError("Sample rates must match for channel merging")

        merged_waveform = merge_channels(waveform_list)
        return ({"waveform": merged_waveform, "sample_rate": sample_rate},)

class AudioChannelSplit(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("LIST[AUDIO]",)
    FUNCTION = "split_channels_node"

    def split_channels_node(self, audio):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        channel_waveforms = split_channels(waveform)
        audio_list = [{"waveform": w, "sample_rate": sample_rate} for w in channel_waveforms]
        return (audio_list,)

class AudioConcatenate(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concatenate_audio_node"

    def concatenate_audio_node(self, audio1, audio2):
        if audio1['sample_rate'] != audio2['sample_rate']:
            raise ValueError("Both audio inputs must have the same sample rate")

        sample_rate = audio1['sample_rate']
        concatenated_waveform = concatenate_audio(audio1['waveform'], audio2['waveform'])
        return ({"waveform": concatenated_waveform, "sample_rate": sample_rate},)

class Audio_Combine(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "weight1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "combine_audio_node"

    def combine_audio_node(self, audio1, audio2, weight1=0.5, weight2=0.5):
        if audio1['sample_rate'] != audio2['sample_rate']:
            raise ValueError("Both audio inputs must have the same sample rate")

        sample_rate = audio1['sample_rate']
        combined_waveform = combine_audio(audio1['waveform'], audio2['waveform'], weight1, weight2)
        return ({"waveform": combined_waveform, "sample_rate": sample_rate},)

class AudioSubtract(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "subtract_audio_node"

    def subtract_audio_node(self, audio1, audio2):
        if audio1['sample_rate'] != audio2['sample_rate']:
            raise ValueError("Both audio inputs must have the same sample rate")

        sample_rate = audio1['sample_rate']
        waveform1 = audio1['waveform']
        waveform2 = audio2['waveform']

        # Ensure both waveforms have the same length
        if waveform1.shape[1] != waveform2.shape[1]:
            max_length = max(waveform1.shape[1], waveform2.shape[1])
            waveform1 = torch.nn.functional.pad(waveform1, (0, max_length - waveform1.shape[1]))
            waveform2 = torch.nn.functional.pad(waveform2, (0, max_length - waveform2.shape[1]))

        subtracted_waveform = waveform1 - waveform2
        return ({"waveform": subtracted_waveform, "sample_rate": sample_rate},)

class AudioInfo(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("duration_seconds", "sample_rate", "num_channels", "num_samples", 
                   "max_amplitude", "mean_amplitude", "rms_amplitude", "bit_depth")
    FUNCTION = "get_audio_info"

    def get_audio_info(self, audio):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        
        # Calculate basic properties
        num_channels = waveform.shape[1]
        num_samples = waveform.shape[2]
        duration_seconds = num_samples / sample_rate
        
        # Calculate amplitude statistics
        max_amplitude = float(torch.max(torch.abs(waveform)))
        mean_amplitude = float(torch.mean(torch.abs(waveform)))
        rms_amplitude = float(torch.sqrt(torch.mean(waveform ** 2)))
        
        # Get bit depth from dtype
        bit_depth = str(waveform.dtype)
        
        return (
            duration_seconds,
            sample_rate,
            num_channels,
            num_samples,
            max_amplitude,
            mean_amplitude,
            rms_amplitude,
            bit_depth
        )

class AudioDither(AudioUtility):
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
    FUNCTION = "dither_audio_node"

    def dither_audio_node(self, audio, bit_depth, noise_shaping):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        quantized_waveform = dither_audio(waveform, bit_depth, noise_shaping)
        return ({"waveform": quantized_waveform, "sample_rate": sample_rate},)