import numpy as np
import torch
import librosa
import cv2
import matplotlib.pyplot as plt

class BaseAudioProcessor:
    def __init__(self, audio, num_frames=None, height=None, width=None):
        self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()  # Convert to mono and numpy array
        self.sample_rate = audio['sample_rate']
        self.num_frames = num_frames
        self.height = height
        self.width = width

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def _enhance_contrast(self, data, power=0.3):
        return np.power(data, power)

    def _resize(self, data, new_width, new_height):
        return cv2.resize(data, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _to_rgb_frames(self, data):
        frames = []
        for i in range(self.num_frames):
            start = i * self.width
            end = (i + 1) * self.width
            frame = data[:, start:end]
            frame_rgb = np.stack([frame] * 3, axis=-1)
            frames.append(frame_rgb)
        return np.stack(frames, axis=0)

class AudioVisualizer(BaseAudioProcessor):
    def create_spectrogram(self):
        print(f"create_spectrogram input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        # Compute the spectrogram
        n_fft = 2048
        hop_length = len(self.audio) // self.num_frames
        S = librosa.stft(self.audio, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        print(f"Spectrogram shape after STFT: {S_db.shape}")

        # Normalize and enhance contrast
        S_db_normalized = self._normalize(S_db)
        S_db_enhanced = self._enhance_contrast(S_db_normalized)

        # Resize and create frames
        S_db_resized = self._resize(S_db_enhanced, self.width * self.num_frames, self.height)
        frames = self._to_rgb_frames(S_db_resized)

        print(f"Final reshaped spectrogram shape: {frames.shape}")

        # Convert to tensor
        tensor = torch.from_numpy(frames).float()

        return tensor

    def create_waveform(self):
        print(f"create_waveform input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        # Compute waveform
        plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
        librosa.display.waveshow(self.audio, sr=self.sample_rate)
        plt.xlim(0, len(self.audio) / self.sample_rate)

        # Save plot to buffer
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.gcf().canvas.draw()
        img = np.array(plt.gcf().canvas.renderer.buffer_rgba())
        plt.close()

        # Convert RGBA to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Normalize image data to [0, 255] and convert to uint8
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

        # Resize image to match desired width and height
        img_resized = self._resize(img_rgb, self.width * self.num_frames, self.height)

        # Create RGB frames
        frames = self._to_rgb_frames(img_resized)

        print(f"Final reshaped waveform shape: {frames.shape}")

        # Convert to tensor
        tensor = torch.from_numpy(frames).float()
        tensor = tensor.permute(0, 3, 1, 2)
        return tensor

    def create_mfcc(self):
        print(f"create_mfcc input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sample_rate)

        # Normalize and enhance contrast
        mfccs_normalized = self._normalize(mfccs)
        mfccs_enhanced = self._enhance_contrast(mfccs_normalized)

        # Resize and create frames
        mfccs_resized = self._resize(mfccs_enhanced, self.width * self.num_frames, self.height)
        frames = self._to_rgb_frames(mfccs_resized)

        print(f"Final reshaped MFCCs shape: {frames.shape}")

        # Convert to tensor
        tensor = torch.from_numpy(frames).float()

        return tensor

    def create_chroma(self):
        print(f"create_chroma input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        # Compute Chroma features
        chroma = librosa.feature.chroma_stft(y=self.audio, sr=self.sample_rate)

        # Normalize and enhance contrast
        chroma_normalized = self._normalize(chroma)
        chroma_enhanced = self._enhance_contrast(chroma_normalized)

        # Resize and create frames
        chroma_resized = self._resize(chroma_enhanced, self.width * self.num_frames, self.height)
        frames = self._to_rgb_frames(chroma_resized)

        print(f"Final reshaped Chroma shape: {frames.shape}")

        # Convert to tensor
        tensor = torch.from_numpy(frames).float()

        return tensor

    def create_tonnetz(self):
        print(f"create_tonnetz input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        # Compute Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=self.audio, sr=self.sample_rate)

        # Normalize and enhance contrast
        tonnetz_normalized = self._normalize(tonnetz)
        tonnetz_enhanced = self._enhance_contrast(tonnetz_normalized)

        # Resize and create frames
        tonnetz_resized = self._resize(tonnetz_enhanced, self.width * self.num_frames, self.height)
        frames = self._to_rgb_frames(tonnetz_resized)

        print(f"Final reshaped Tonnetz shape: {frames.shape}")

        # Convert to tensor
        tensor = torch.from_numpy(frames).float()

        return tensor

    def create_spectral_centroid(self):
        print(f"create_spectral_centroid input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        # Compute Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sample_rate)

        # Normalize and enhance contrast
        spectral_centroid_normalized = self._normalize(spectral_centroid)
        spectral_centroid_enhanced = self._enhance_contrast(spectral_centroid_normalized)

        # Resize and create frames
        spectral_centroid_resized = self._resize(spectral_centroid_enhanced, self.width * self.num_frames, self.height)
        frames = self._to_rgb_frames(spectral_centroid_resized)

        print(f"Final reshaped Spectral Centroid shape: {frames.shape}")

        # Convert to tensor
        tensor = torch.from_numpy(frames).float()

        return tensor

class AudioFeatureExtractor(BaseAudioProcessor):
    def __init__(self, audio, feature_type='amplitude_envelope'):
        super().__init__(audio)
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

    def _amplitude_envelope(self):
        return np.abs(self.audio)

    def _rms_energy(self):
        return np.sqrt(np.mean(self.audio**2))

    def _spectral_centroid(self):
        S, _ = librosa.magphase(librosa.stft(self.audio))
        return librosa.feature.spectral_centroid(S=S, sr=self.sample_rate)[0]

    def _onset_detection(self):
        return librosa.onset.onset_strength(y=self.audio, sr=self.sample_rate)

    def _chroma_features(self):
        return librosa.feature.chroma_stft(y=self.audio, sr=self.sample_rate)

