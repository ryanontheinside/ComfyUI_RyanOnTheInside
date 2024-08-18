import torch
import numpy as np
import librosa
import openunmix
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import cv2
from .audio_utils import AudioVisualizer

class AudioSeparator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "AUDIO", "AUDIO", "AUDIO","AUDIO")
    RETURN_NAMES = ("audio", "video_frames","drums_audio", "vocals_audio", "bass_audio", "other_audio")
    FUNCTION = "process_audio"

    def __init__(self):
        self.separator = openunmix.umxl(targets=['drums', 'vocals', 'bass', 'other'], device='cpu')

    def process_audio(self, audio, video_frames):
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        # Extract dimensions from video_frames
        num_frames, height, width, _ = video_frames.shape
        print(f"Video frames shape: {video_frames.shape}")

        # Ensure the waveform is in the correct shape (nb_channels, nb_samples)
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)  # Remove batch dimension if present
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if mono
        if waveform.shape[0] != 2:
            waveform = waveform.repeat(2, 1)  # Duplicate mono to stereo if necessary
        print(f"Waveform shape after preprocessing: {waveform.shape}")

        # Calculate the expected audio length based on video frames
        expected_audio_length = int(num_frames * sample_rate / 30)  # Assuming 30 fps
        
        # Trim or pad the audio to match the expected length
        if waveform.shape[1] > expected_audio_length:
            waveform = waveform[:, :expected_audio_length]
        elif waveform.shape[1] < expected_audio_length:
            pad_length = expected_audio_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        print(f"Waveform shape after adjusting length: {waveform.shape}")

        # Add a batch dimension
        waveform = waveform.unsqueeze(0)
        print(f"Waveform shape before separation: {waveform.shape}")

        # Separate the audio using Open-Unmix
        estimates = self.separator(waveform)
        print(f"Estimates shape: {estimates.shape}")

        # Create isolated audio objects and spectrograms for each target
        isolated_audio = {}
        spectrograms = {}
        for i, target in enumerate(['drums', 'vocals', 'bass', 'other']):
            target_waveform = estimates[:, i, :, :]  # Shape: (1, 2, num_samples)
            print(f"{target} audio shape: {target_waveform.shape}")
            
            isolated_audio[target] = {
                'waveform': target_waveform,
                'sample_rate': sample_rate
            }


        return (
            audio,
            video_frames,
            isolated_audio['drums'],
            isolated_audio['vocals'],
            isolated_audio['bass'],
            isolated_audio['other'],
        )
    
class AudioFeatureVisualizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
                "visualization_type": (["waveform", "spectrogram", "mfcc", "chroma", "tonnetz", "spectral_centroid"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_audio_feature"
    CATEGORY = "audio"

    def visualize_audio_feature(self, audio, video_frames, visualization_type):
        # Extract dimensions from video_frames
        num_frames, height, width, _ = video_frames.shape
        print(f"Video frames shape: {video_frames.shape}")

        visualizer = AudioVisualizer(audio, num_frames, height, width)
        
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

        print(f"Output mask shape: {mask.shape}")

        return (mask,)