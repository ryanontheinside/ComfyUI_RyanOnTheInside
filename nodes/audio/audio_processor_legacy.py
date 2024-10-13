import numpy as np
import torch
import librosa
import cv2
import matplotlib.pyplot as plt

import numpy as np
import librosa
import torch
import cv2
import matplotlib.pyplot as plt




class BaseAudioProcessor:
    def __init__(self, audio, num_frames, height, width, frame_rate):
        self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()  # Convert to mono and numpy array
        self.sample_rate = audio['sample_rate']
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_rate = frame_rate
        
        self.audio_duration = len(self.audio) / self.sample_rate
        self.frame_duration = 1 / self.frame_rate if self.frame_rate > 0 else self.audio_duration / self.num_frames

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def _enhance_contrast(self, data, power=0.3):
        return np.power(data, power)

    def _resize(self, data, new_width, new_height):
        return cv2.resize(data, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = (frame_index + 1) * self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        return self.audio[start_sample:end_sample]

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

class AudioVisualizer(BaseAudioProcessor):  

    def __init__(self, audio, num_frames, height, width, frame_rate, x_axis, y_axis, cmap):
        super().__init__(audio, num_frames, height, width, frame_rate)
        plt.ioff()  # Turn off interactive mode
        self.x_axis = x_axis if x_axis != 'None' else None
        self.y_axis = y_axis if y_axis != 'None' else None
        self.cmap = cmap

    def _generate_frame(self, data):
        """Helper method to create frames using librosa's specshow with better visual appeal."""
        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
        librosa.display.specshow(data, sr=self.sample_rate, x_axis=self.x_axis, y_axis=self.y_axis, cmap=self.cmap, ax=ax)
        if self.x_axis == 'off':
            ax.set_xticks([])
        if self.y_axis == 'off':
            ax.set_yticks([])
        plt.tight_layout(pad=0)
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(self.height, self.width, 3)
        
        plt.close(fig)
        return frame

    def create_spectrogram(self):
        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            n_fft = min(2048, len(audio_frame))
            S = librosa.stft(audio_frame, n_fft=n_fft)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            frame = self._generate_frame(S_db)
            frames.append(frame)

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_waveform(self):
        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            
            fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
            librosa.display.waveshow(audio_frame, sr=self.sample_rate, ax=ax, x_axis=self.x_axis)
            if self.x_axis == 'off':
                ax.set_xticks([])
            if self.y_axis == 'off':
                ax.set_yticks([])
            plt.tight_layout(pad=0)
            
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(self.height, self.width, 3)
            frames.append(frame)
            plt.close(fig)

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_mfcc(self):
        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            mfccs = librosa.feature.mfcc(y=audio_frame, sr=self.sample_rate, n_mfcc=20)
            frame = self._generate_frame(mfccs)
            frames.append(frame)

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_chroma(self):
        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            chroma = librosa.feature.chroma_stft(y=audio_frame, sr=self.sample_rate)
            frame = self._generate_frame(chroma)
            frames.append(frame)

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_tonnetz(self):
        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            tonnetz = librosa.feature.tonnetz(y=audio_frame, sr=self.sample_rate)
            frame = self._generate_frame(tonnetz)
            frames.append(frame)

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_spectral_centroid(self):
        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            centroid = librosa.feature.spectral_centroid(y=audio_frame, sr=self.sample_rate)
            frame = self._generate_frame(centroid)
            frames.append(frame)

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0
    
# class AudioVisualizer(BaseAudioProcessor):  
#     def create_spectrogram(self):
#         frames = []
#         for i in range(self.num_frames):
#             audio_frame = self._get_audio_frame(i)
            
#             n_fft = min(2048, len(audio_frame))
#             S = librosa.stft(audio_frame, n_fft=n_fft)
#             S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
#             S_db_normalized = self._normalize(S_db)
#             S_db_enhanced = self._enhance_contrast(S_db_normalized)
            
#             S_db_resized = self._resize(S_db_enhanced, self.width, self.height)
#             S_db_resized = (S_db_resized * 255).astype(np.uint8)
#             S_db_resized = np.repeat(S_db_resized[:, :, np.newaxis], 3, axis=2)  # Convert to RGB
#             frames.append(S_db_resized)

#         frames = np.stack(frames, axis=0)  # Shape: (B, H, W, C)
#         return torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]

#     def create_waveform(self):

#         frames = []
#         for i in range(self.num_frames):
#             audio_frame = self._get_audio_frame(i)
            
#             plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
#             plt.plot(audio_frame)
#             plt.axis('off')
#             plt.tight_layout(pad=0)
            
#             plt.gcf().canvas.draw()
#             frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
#             frame = frame.reshape(self.height, self.width, 3)
#             frames.append(frame)
#             plt.close()

#         frames = np.stack(frames, axis=0)

#         return torch.from_numpy(frames).float() / 255.0

#     def create_mfcc(self):
#         frames = []
#         for i in range(self.num_frames):
#             audio_frame = self._get_audio_frame(i)

#             mfccs = librosa.feature.mfcc(y=audio_frame, sr=self.sample_rate, n_mfcc=20)
#             mfccs_normalized = self._normalize(mfccs)
#             mfccs_enhanced = self._enhance_contrast(mfccs_normalized)

#             mfccs_transposed = mfccs_enhanced.T
#             mfccs_resized = self._resize(mfccs_transposed, self.width, self.height)
#             mfccs_resized = (mfccs_resized * 255).astype(np.uint8)
#             mfccs_resized = np.repeat(mfccs_resized[:, :, np.newaxis], 3, axis=2)  # Convert to RGB
#             frames.append(mfccs_resized)

#         frames = np.stack(frames, axis=0)
#         return torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]

#     def create_chroma(self):
#         frames = []
#         for i in range(self.num_frames):
#             audio_frame = self._get_audio_frame(i)

#             chroma = librosa.feature.chroma_stft(y=audio_frame, sr=self.sample_rate)
#             chroma_normalized = self._normalize(chroma)
#             chroma_enhanced = self._enhance_contrast(chroma_normalized)

#             chroma_transposed = chroma_enhanced.T
#             chroma_resized = self._resize(chroma_transposed, self.width, self.height)
#             chroma_resized = (chroma_resized * 255).astype(np.uint8)
#             chroma_resized = np.repeat(chroma_resized[:, :, np.newaxis], 3, axis=2)  # Convert to RGB
#             frames.append(chroma_resized)

#         frames = np.stack(frames, axis=0)
#         return torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]

#     def create_tonnetz(self):
#         frames = []
#         for i in range(self.num_frames):
#             audio_frame = self._get_audio_frame(i)

#             tonnetz = librosa.feature.tonnetz(y=audio_frame, sr=self.sample_rate)
#             tonnetz_normalized = self._normalize(tonnetz)
#             tonnetz_enhanced = self._enhance_contrast(tonnetz_normalized)

#             tonnetz_transposed = tonnetz_enhanced.T
#             tonnetz_resized = self._resize(tonnetz_transposed, self.width, self.height)
#             tonnetz_resized = (tonnetz_resized * 255).astype(np.uint8)
#             tonnetz_resized = np.repeat(tonnetz_resized[:, :, np.newaxis], 3, axis=2)  # Convert to RGB
#             frames.append(tonnetz_resized)

#         frames = np.stack(frames, axis=0)
#         return torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]

#     def create_spectral_centroid(self):
#         frames = []
#         for i in range(self.num_frames):
#             audio_frame = self._get_audio_frame(i)

#             centroid = librosa.feature.spectral_centroid(y=audio_frame, sr=self.sample_rate)
#             centroid_normalized = self._normalize(centroid)
#             centroid_enhanced = self._enhance_contrast(centroid_normalized)

#             centroid_transposed = centroid_enhanced.T
#             centroid_resized = self._resize(centroid_transposed, self.width, self.height)
#             centroid_resized = (centroid_resized * 255).astype(np.uint8)
#             centroid_resized = np.repeat(centroid_resized[:, :, np.newaxis], 3, axis=2)  # Convert to RGB
#             frames.append(centroid_resized)

#         frames = np.stack(frames, axis=0)
#         return torch.from_numpy(frames).float() / 255.0  # Normalize to [0, 1]

class AudioFeatureExtractor(BaseAudioProcessor):
    def __init__(self, audio, num_frames, frame_rate, feature_type='amplitude_envelope'):
        super().__init__(audio, num_frames, None, None, frame_rate)
        self.feature_type = feature_type

    def extract(self):
        if self.feature_type == 'amplitude_envelope':
            return self._amplitude_envelope()
        elif self.feature_type == 'rms_energy':
            return self._rms_energy()
        elif self.feature_type == 'spectral_centroid':
            return self._spectral_centroid()
        elif self.feature_type == 'onset_detection':
            return self._onset_detection()
        elif self.feature_type == 'chroma_features':
            return self._chroma_features()
        else:
            raise ValueError("Unsupported feature type")

#TODO HANDLE NO MASKS
#TODO HANDLE FRAME MASK COUNT MISMATCH
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
