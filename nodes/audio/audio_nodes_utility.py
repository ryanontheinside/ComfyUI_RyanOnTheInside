from .audio_nodes import AudioNodeBase
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn.functional as nnf  # Added torch.nn.functional
import librosa

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
    FUNCTION = "pad_audio"

    def pad_audio(self, audio, pad_left, pad_right, pad_mode):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        # Use torch.nn.functional.pad for padding
        padded_waveform = nnf.pad(waveform, (pad_left, pad_right), mode=pad_mode)
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
    FUNCTION = "normalize_volume"

    def normalize_volume(self, audio, target_level):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        
        # Calculate current RMS level in dB
        rms = torch.sqrt(torch.mean(waveform ** 2))
        current_db = 20 * torch.log10(rms + 1e-6)  # Add small value to avoid log(0)
        
        # Calculate the required gain in dB
        gain_db = target_level - current_db.item()
        gain = 10 ** (gain_db / 20)
        
        # Apply gain
        normalized_waveform = waveform * gain
        
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
    FUNCTION = "resample_audio"

    def resample_audio(self, audio, new_sample_rate):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        resampler = T.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        resampled_waveform = resampler(waveform)
        return ({"waveform": resampled_waveform, "sample_rate": new_sample_rate},)

class AudioChannelMerge(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "merge_channels"

    def merge_channels(self, audio1, audio2):
        waveform1, sample_rate1 = audio1['waveform'], audio1['sample_rate']
        waveform2, sample_rate2 = audio2['waveform'], audio2['sample_rate']
        
        if sample_rate1 != sample_rate2:
            raise ValueError("Sample rates must match for channel merging")
        
        # Ensure both waveforms have the same length
        min_length = min(waveform1.shape[-1], waveform2.shape[-1])
        waveform1 = waveform1[..., :min_length]
        waveform2 = waveform2[..., :min_length]
        
        merged_waveform = torch.cat([waveform1, waveform2], dim=0)
        return ({"waveform": merged_waveform, "sample_rate": sample_rate1},)

class AudioChannelSplit(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    FUNCTION = "split_channels"

    def split_channels(self, audio):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        
        if waveform.shape[0] < 2:
            raise ValueError("Input audio must have at least 2 channels for splitting")
        
        # Split into individual channels
        channel_waveforms = []
        for i in range(waveform.shape[0]):
            channel_waveforms.append({"waveform": waveform[i:i+1, :], "sample_rate": sample_rate})
        
        return tuple(channel_waveforms)



    

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
    FUNCTION = "concatenate_audio"

    def concatenate_audio(self, audio1, audio2):
        # Ensure both audio inputs have the same sample rate
        if audio1['sample_rate'] != audio2['sample_rate']:
            raise ValueError("Both audio inputs must have the same sample rate")

        sample_rate = audio1['sample_rate']

        # Concatenate waveforms
        concatenated_waveform = torch.cat([audio1['waveform'], audio2['waveform']], dim=-1)

        return ({"waveform": concatenated_waveform, "sample_rate": sample_rate},)

class AudioCombine(AudioUtility):
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
    FUNCTION = "combine_audio"

    def combine_audio(self, audio1, audio2, weight1=1.0, weight2=1.0):
        # Ensure both audio inputs have the same sample rate
        if audio1['sample_rate'] != audio2['sample_rate']:
            raise ValueError("Both audio inputs must have the same sample rate")

        sample_rate = audio1['sample_rate']

        # Get the maximum length of the two audio inputs
        max_length = max(audio1['waveform'].shape[-1], audio2['waveform'].shape[-1])

        # Pad shorter audio to match the longest one
        waveform1 = audio1['waveform']
        waveform2 = audio2['waveform']

        if waveform1.shape[-1] < max_length:
            pad_length = max_length - waveform1.shape[-1]
            waveform1 = torch.nn.functional.pad(waveform1, (0, pad_length))

        if waveform2.shape[-1] < max_length:
            pad_length = max_length - waveform2.shape[-1]
            waveform2 = torch.nn.functional.pad(waveform2, (0, pad_length))

        # Combine waveforms
        combined_waveform = (waveform1 * weight1) + (waveform2 * weight2)

        # Normalize the combined waveform
        max_amplitude = torch.max(torch.abs(combined_waveform))
        if max_amplitude > 1.0:
            combined_waveform = combined_waveform / max_amplitude

        return ({"waveform": combined_waveform, "sample_rate": sample_rate},)
