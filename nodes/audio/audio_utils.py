import numpy as np
import torch
import librosa
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

class AudioVisualizer(BaseAudioProcessor):  
    def create_spectrogram(self):
        print(f"create_spectrogram input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            
            n_fft = min(2048, len(audio_frame))
            S = librosa.stft(audio_frame, n_fft=n_fft)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            S_db_normalized = self._normalize(S_db)
            S_db_enhanced = self._enhance_contrast(S_db_normalized)
            
            S_db_resized = self._resize(S_db_enhanced, self.width, self.height)
            frames.append(S_db_resized)

        frames = np.stack(frames, axis=0)
        print(f"Final reshaped spectrogram shape: {frames.shape}")

        return torch.from_numpy(frames).float()

    def create_waveform(self):
        print(f"create_waveform input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            
            plt.figure(figsize=(self.width / 100, self.height / 100), dpi=100)
            plt.plot(audio_frame)
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            plt.gcf().canvas.draw()
            frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(self.height, self.width, 3)
            frames.append(frame)
            plt.close()

        frames = np.stack(frames, axis=0)
        print(f"Final reshaped waveform shape: {frames.shape}")

        return torch.from_numpy(frames).float() / 255.0

    def create_mfcc(self):
        print(f"create_mfcc input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            
            mfccs = librosa.feature.mfcc(y=audio_frame, sr=self.sample_rate, n_mfcc=self.height)
            mfccs_normalized = self._normalize(mfccs)
            mfccs_enhanced = self._enhance_contrast(mfccs_normalized)
            
            mfccs_resized = self._resize(mfccs_enhanced, self.width, self.height)
            frames.append(mfccs_resized)

        frames = np.stack(frames, axis=0)
        print(f"Final reshaped MFCCs shape: {frames.shape}")

        return torch.from_numpy(frames).float()

    def create_chroma(self):
        print(f"create_chroma input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            
            chroma = librosa.feature.chroma_stft(y=audio_frame, sr=self.sample_rate)
            chroma_normalized = self._normalize(chroma)
            chroma_enhanced = self._enhance_contrast(chroma_normalized)
            
            chroma_resized = self._resize(chroma_enhanced, self.width, self.height)
            frames.append(chroma_resized)

        frames = np.stack(frames, axis=0)
        print(f"Final reshaped Chroma shape: {frames.shape}")

        return torch.from_numpy(frames).float()

    def create_tonnetz(self):
        print(f"create_tonnetz input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            
            tonnetz = librosa.feature.tonnetz(y=audio_frame, sr=self.sample_rate)
            tonnetz_normalized = self._normalize(tonnetz)
            tonnetz_enhanced = self._enhance_contrast(tonnetz_normalized)
            
            tonnetz_resized = self._resize(tonnetz_enhanced, self.width, self.height)
            frames.append(tonnetz_resized)

        frames = np.stack(frames, axis=0)
        print(f"Final reshaped Tonnetz shape: {frames.shape}")

        return torch.from_numpy(frames).float()

    def create_spectral_centroid(self):
        print(f"create_spectral_centroid input shapes: audio={self.audio.shape}, num_frames={self.num_frames}, height={self.height}, width={self.width}")

        frames = []
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            
            centroid = librosa.feature.spectral_centroid(y=audio_frame, sr=self.sample_rate)
            centroid_normalized = self._normalize(centroid)
            centroid_enhanced = self._enhance_contrast(centroid_normalized)
            
            centroid_resized = self._resize(centroid_enhanced, self.width, self.height)
            frames.append(centroid_resized)

        frames = np.stack(frames, axis=0)
        print(f"Final reshaped Spectral Centroid shape: {frames.shape}")

        return torch.from_numpy(frames).float()

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