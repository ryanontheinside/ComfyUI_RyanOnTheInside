import torch
import numpy as np
import librosa
import openunmix
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2
from .audio_utils import AudioVisualizer
from scipy import signal
import logging

from ... import RyanOnTheInside
from ..flex.feature_pipe import FeaturePipe
from ... import RyanOnTheInside



class AudioNodeBase(RyanOnTheInside):
    CATEGORY= "RyanOnTheInside/Audio"

class AudioSeparator(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "FEATURE_PIPE")
    RETURN_NAMES = ("audio", "drums_audio", "vocals_audio", "bass_audio", "other_audio", "feature_pipe")
    FUNCTION = "process_audio"

    def __init__(self):
        self.separator = openunmix.umxl(targets=['drums', 'vocals', 'bass', 'other'], device='cpu')

    def process_audio(self, audio, video_frames, frame_rate):
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        num_frames, height, width, _ = video_frames.shape

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # Remove batch dimension if present
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if mono
        if waveform.shape[0] != 2:
            waveform = waveform.repeat(2, 1)  # Duplicate mono to stereo if necessary
            
        waveform = waveform.unsqueeze(0)

        # Separate the audio using Open-Unmix
        estimates = self.separator(waveform)

        # Create isolated audio objects for each target
        isolated_audio = {}
        for i, target in enumerate(['drums', 'vocals', 'bass', 'other']):
            target_waveform = estimates[:, i, :, :]  # Shape: (1, 2, num_samples)
            
            isolated_audio[target] = {
                'waveform': target_waveform,
                'sample_rate': sample_rate,
                'frame_rate': frame_rate
            }

        # Create FeaturePipe
        feature_pipe = FeaturePipe(frame_rate, video_frames)

        return (
            audio,
            isolated_audio['drums'],
            isolated_audio['vocals'],
            isolated_audio['bass'],
            isolated_audio['other'],
            feature_pipe,
        )

class AudioFilter(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filters": ("FREQUENCY_FILTER",),
                "descrip": ("STRING", {"default":cls.DESCRIPTION})
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_filters"
    DESCRIPTION = """Applies frequency filters to audio:
- `audio`: Input audio to be filtered
- `filters`: Frequency filters to be applied (FREQUENCY_FILTER type)"""

    def apply_filters(self, audio, filters):
        audio_np = audio['waveform'].cpu().numpy().squeeze(0)
        sample_rate = audio['sample_rate']

        filtered_channels = []
        for channel in audio_np:
            filtered_channel = channel
            for filter_params in filters:
                filtered_channel = self.apply_single_filter(filtered_channel, filter_params, sample_rate)
            filtered_channels.append(filtered_channel)

        filtered_audio = np.stack(filtered_channels)

        max_val = np.max(np.abs(filtered_audio))
        if max_val > 1.0:
            filtered_audio = filtered_audio / max_val

        filtered_tensor = torch.from_numpy(filtered_audio).unsqueeze(0).float()

        return ({"waveform": filtered_tensor, "sample_rate": sample_rate},)

    def apply_single_filter(self, audio, filter_params, sample_rate):
        filter_type = filter_params['type']
        order = filter_params['order']
        cutoff = filter_params['cutoff']

        if filter_type == 'lowpass':
            b, a = signal.butter(order, cutoff, btype='low', fs=sample_rate)
        elif filter_type == 'highpass':
            b, a = signal.butter(order, cutoff, btype='high', fs=sample_rate)
        elif filter_type == 'bandpass':
            low = cutoff - 50
            high = cutoff + 50
            b, a = signal.butter(order, [low, high], btype='band', fs=sample_rate)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        return signal.lfilter(b, a, audio)

class FrequencyFilterPreset(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([
                    "isolate_kick_drum",
                    "isolate_snare_drum",
                    "isolate_hi_hat",
                    "isolate_bass",
                    "isolate_vocals",
                    "remove_rumble",
                    "brighten_mix",
                    "remove_hiss"
                ],),
            },
            "optional": {
                "previous_filter": ("FREQUENCY_FILTER",),
            },
        }

    RETURN_TYPES = ("FREQUENCY_FILTER",)
    FUNCTION = "create_preset_filter_chain"

    def create_preset_filter_chain(self, preset, previous_filter=None):
        new_filters = self.get_preset_filters(preset)

        if previous_filter:
            if isinstance(previous_filter, list):
                return (previous_filter + new_filters,)
            else:
                return ([previous_filter] + new_filters,)
        else:
            return (new_filters,)

    def get_preset_filters(self, preset):
        if preset == "isolate_kick_drum":
            return [
                {"type": "highpass", "order": 2, "cutoff": 30},
                {"type": "lowpass", "order": 4, "cutoff": 200}
            ]
        elif preset == "isolate_snare_drum":
            return [
                {"type": "bandpass", "order": 4, "cutoff": 200},
                {"type": "bandpass", "order": 4, "cutoff": 5000}
            ]
        elif preset == "isolate_hi_hat":
            return [
                {"type": "highpass", "order": 4, "cutoff": 8000}
            ]
        elif preset == "isolate_bass":
            return [
                {"type": "lowpass", "order": 4, "cutoff": 250}
            ]
        elif preset == "isolate_vocals":
            return [
                {"type": "bandpass", "order": 2, "cutoff": 300},
                {"type": "bandpass", "order": 2, "cutoff": 4000}
            ]
        elif preset == "remove_rumble":
            return [
                {"type": "highpass", "order": 4, "cutoff": 40}
            ]
        elif preset == "brighten_mix":
            return [
                {"type": "highpass", "order": 2, "cutoff": 6000}
            ]
        elif preset == "remove_hiss":
            return [
                {"type": "lowpass", "order": 6, "cutoff": 7500}
            ]
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
class FrequencyFilterCustom(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filter_type": (["lowpass", "highpass", "bandpass"],),
                "order": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "cutoff": ("FLOAT", {"default": 1000, "min": 20, "max": 20000, "step": 1}),
            },
            "optional": {
                "previous_filter": ("FREQUENCY_FILTER",),
            },
        }

    RETURN_TYPES = ("FREQUENCY_FILTER",)
    FUNCTION = "create_filter"

    def create_filter(self, filter_type, order, cutoff, previous_filter=None):
        filter_params = {
            "type": filter_type,
            "order": order,
            "cutoff": cutoff,
        }

        if previous_filter:
            if isinstance(previous_filter, list):
                previous_filter.append(filter_params)
                return (previous_filter,)
            else:
                return ([previous_filter, filter_params],)
        else:
            return ([filter_params],)

class AudioFeatureVisualizer(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
                "visualization_type": ([
                    "waveform", 
                    "spectrogram", 
                    "mfcc", 
                    "chroma", 
                    "tonnetz", 
                    "spectral_centroid"
                    ],),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_audio_feature"
    CATEGORY = "audio"

    def visualize_audio_feature(self, audio, video_frames, visualization_type, frame_rate):
        num_frames, height, width, _ = video_frames.shape

        visualizer = AudioVisualizer(audio, num_frames, height, width, frame_rate)
        
        if visualization_type == "waveform":
            mask = visualizer.create_waveform()
        elif visualization_type == "spectrogram":
            mask = visualizer.create_spectrogram()
        elif visualization_type == "mfcc":
            mask = visualizer.create_mfcc()
        elif visualization_type == "chroma":
            mask = visualizer.create_chroma()
        elif visualization_type == "tonnetz":
            mask = visualizer.create_tonnetz()
        elif visualization_type == "spectral_centroid":
            mask = visualizer.create_spectral_centroid()
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")


        return (mask,)