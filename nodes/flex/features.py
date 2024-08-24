from abc import ABC, abstractmethod
import numpy as np
import librosa
from scipy import ndimage
import torch
import torch.nn.functional as F


class BaseFeature(ABC):
    def __init__(self, name, feature_type, frame_rate, frame_count):
        self.name = name
        self.type = feature_type
        self.frame_rate = frame_rate
        self.frame_count = frame_count
        self.data = None

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def get_value_at_frame(self, frame_index):
        pass

    def normalize(self):
        if self.data is not None:
            self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return self

class TimeFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, effect_type='smooth', speed=1.0, offset=0.0):
        super().__init__(name, "time", frame_rate, frame_count)
        self.effect_type = effect_type
        self.speed = speed
        self.offset = offset

    def extract(self):
        t = np.linspace(0, self.frame_count / self.frame_rate, self.frame_count)
        t = (t * self.speed + self.offset) % 1

        if self.effect_type == 'smooth':
            self.data = t
        elif self.effect_type == 'accelerate':
            self.data = t ** 2
        elif self.effect_type == 'pulse':
            self.data = (np.sin(2 * np.pi * t) + 1) / 2
        elif self.effect_type == 'sawtooth':
            self.data = t
        elif self.effect_type == 'bounce':
            self.data = 1 - np.abs(1 - 2 * t)
        else:
            raise ValueError("Unsupported effect type")
        
        return self.normalize()

    def get_value_at_frame(self, frame_index):
        if self.data is None:
            self.extract()
        return self.data[frame_index]  
    
class AudioFeature(BaseFeature):
    def __init__(self, name, audio, num_frames, frame_rate, feature_type='amplitude_envelope'):
        super().__init__(name, "audio", frame_rate,num_frames)
        self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()
        self.sample_rate = audio['sample_rate']
        self.num_frames = num_frames
        self.feature_type = feature_type
        self.frame_duration = 1 / self.frame_rate if self.frame_rate > 0 else len(self.audio) / (self.sample_rate * self.num_frames)

    def extract(self):
        if self.feature_type == 'amplitude_envelope':
            self.data = self._amplitude_envelope()
        elif self.feature_type == 'rms_energy':
            self.data = self._rms_energy()
        elif self.feature_type == 'spectral_centroid':
            self.data = self._spectral_centroid()
        elif self.feature_type == 'onset_detection':
            self.data = self._onset_detection()
        elif self.feature_type == 'chroma_features':
            self.data = self._chroma_features()
        else:
            raise ValueError("Unsupported feature type")
        return self.normalize()

    def get_value_at_frame(self, frame_index):
        if self.data is None:
            self.extract()
        return self.data[frame_index]

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = (frame_index + 1) * self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        return self.audio[start_sample:end_sample]

    def _amplitude_envelope(self):
        return np.array([np.max(np.abs(self._get_audio_frame(i))) for i in range(self.num_frames)])

    def _rms_energy(self):
        return np.array([np.sqrt(np.mean(self._get_audio_frame(i)**2)) for i in range(self.num_frames)])

    def _spectral_centroid(self):
        return np.array([np.mean(librosa.feature.spectral_centroid(y=self._get_audio_frame(i), sr=self.sample_rate)[0]) for i in range(self.num_frames)])

    def _onset_detection(self):
        return np.array([np.mean(librosa.onset.onset_strength(y=self._get_audio_frame(i), sr=self.sample_rate)) for i in range(self.num_frames)])

    def _chroma_features(self):
        return np.array([np.mean(librosa.feature.chroma_stft(y=self._get_audio_frame(i), sr=self.sample_rate), axis=1) for i in range(self.num_frames)])

class DepthFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, depth_maps, feature_name='mean_depth'):
        super().__init__(name, "depth", frame_rate, frame_count)
        self.depth_maps = depth_maps  # Expected shape: (B, H, W, C)
        self.features = None
        self.feature_name = feature_name
        self.available_features = [
            "mean_depth", "depth_variance", "depth_range", "gradient_magnitude",
            "foreground_ratio", "midground_ratio", "background_ratio"
        ]

    def extract(self):
        self.features = {feature: [] for feature in self.available_features}

        combined_depth = torch.mean(self.depth_maps, dim=-1)  

        for depth_map in combined_depth:
            depth_min = torch.min(depth_map)
            depth_max = torch.max(depth_map)
            if depth_max > depth_min:
                normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                normalized_depth = torch.zeros_like(depth_map)
            
            self.features['mean_depth'].append(torch.mean(normalized_depth).item())
            self.features['depth_variance'].append(torch.var(normalized_depth).item())
            self.features['depth_range'].append((depth_max - depth_min).item())
            
            grad_y, grad_x = torch.gradient(normalized_depth)
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            self.features['gradient_magnitude'].append(torch.mean(gradient_magnitude).item())
            
            total_pixels = normalized_depth.numel()
            foreground = torch.sum(normalized_depth < 0.33).item() / total_pixels
            midground = torch.sum((normalized_depth >= 0.33) & (normalized_depth < 0.66)).item() / total_pixels
            background = torch.sum(normalized_depth >= 0.66).item() / total_pixels
            self.features['foreground_ratio'].append(foreground)
            self.features['midground_ratio'].append(midground)
            self.features['background_ratio'].append(background)

        for key in self.features:
            feature_tensor = torch.tensor(self.features[key])
            feature_min = torch.min(feature_tensor)
            feature_max = torch.max(feature_tensor)
            if feature_max > feature_min:
                self.features[key] = ((feature_tensor - feature_min) / (feature_max - feature_min)).tolist()
            else:
                self.features[key] = torch.zeros_like(feature_tensor).tolist()

        return self

    def get_value_at_frame(self, frame_index):
        if self.features is None:
            self.extract()
        return self.features[self.feature_name][frame_index]

    def get_feature_sequence(self, feature_name=None):
        if self.features is None:
            self.extract()
        if feature_name is None:
            feature_name = self.feature_name
        return self.features.get(feature_name, None)

    def set_active_feature(self, feature_name):
        if feature_name in self.available_features:
            self.feature_name = feature_name
        else:
            raise ValueError(f"Invalid feature name. Available features are: {', '.join(self.available_features)}")