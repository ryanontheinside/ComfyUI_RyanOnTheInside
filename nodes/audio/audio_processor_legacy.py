import numpy as np
import torch
import librosa
import cv2
import matplotlib.pyplot as plt
from comfy.utils import ProgressBar
import numpy as np
import librosa
import torch
import cv2
import matplotlib.pyplot as plt
from ... import ProgressMixin




class BaseAudioProcessor(ProgressMixin):
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
        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
        librosa.display.specshow(data, sr=self.sample_rate, x_axis=self.x_axis, y_axis=self.y_axis, cmap=self.cmap, ax=ax)
        if self.x_axis == 'off':
            ax.set_xticks([])
        if self.y_axis == 'off':
            ax.set_yticks([])
        plt.tight_layout(pad=0)
        
        fig.canvas.draw()
        
        # Get the actual dimensions of the rendered figure
        width, height = fig.canvas.get_width_height()
        
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(height, width, 3)
        
        # Resize the frame to match the desired dimensions
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        plt.close(fig)
        return frame

    def create_spectrogram(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Spectrogram")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            n_fft = min(2048, len(audio_frame))
            S = librosa.stft(audio_frame, n_fft=n_fft)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            frame = self._generate_frame(S_db)
            frames.append(frame)
            self.update_progress()
        self.end_progress()

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_waveform(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Waveform")
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
            
            # Get the actual dimensions of the rendered figure
            width, height = fig.canvas.get_width_height()
            
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(height, width, 3)
            
            # Resize the frame to match the desired dimensions
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            
            frames.append(frame)
            plt.close(fig)
            self.update_progress()
        self.end_progress()

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_mfcc(self):
        frames = []
        self.start_progress(self.num_frames, "Creating MFCC")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            mfccs = librosa.feature.mfcc(y=audio_frame, sr=self.sample_rate, n_mfcc=20)
            frame = self._generate_frame(mfccs)
            frames.append(frame)
            self.update_progress()
        self.end_progress()

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_chroma(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Chroma")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            chroma = librosa.feature.chroma_stft(y=audio_frame, sr=self.sample_rate)
            frame = self._generate_frame(chroma)
            frames.append(frame)
            self.update_progress()
        self.end_progress()

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_tonnetz(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Tonnetz")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            tonnetz = librosa.feature.tonnetz(y=audio_frame, sr=self.sample_rate)
            frame = self._generate_frame(tonnetz)
            frames.append(frame)
            self.update_progress()
        self.end_progress()

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0

    def create_spectral_centroid(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Spectral Centroid")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            centroid = librosa.feature.spectral_centroid(y=audio_frame, sr=self.sample_rate)
            frame = self._generate_frame(centroid)
            frames.append(frame)
            self.update_progress()
        self.end_progress()

        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float() / 255.0
    

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

import numpy as np
import torch
import pygame
import librosa
from comfy.utils import ProgressBar

class PygameAudioVisualizer(BaseAudioProcessor):
    def __init__(self, audio, num_frames, height, width, frame_rate, scroll_direction='left'):
        super().__init__(audio, num_frames, height, width, frame_rate)
        self.height = height
        self.width = width
        self.scroll_direction = scroll_direction.lower()
        pygame.init()
        # Create a Pygame surface for drawing
        self.screen = pygame.Surface((self.width, self.height))
        # Set up font for text rendering (if needed)
        self.font = pygame.font.SysFont(None, 24)
        # Precompute the full waveform
        self.full_waveform = self.audio / np.max(np.abs(self.audio)) if np.max(np.abs(self.audio)) != 0 else self.audio
        self.total_samples = len(self.full_waveform)

    def _surface_to_array(self, surface):
        # Convert Pygame surface to NumPy array
        frame_array = pygame.surfarray.array3d(surface)
        # Transpose to (H, W, C)
        frame_array = np.transpose(frame_array, (1, 0, 2))
        return frame_array

    def create_waveform(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Scrolling Waveform with Pygame")
        samples_per_frame = int(self.total_samples / self.num_frames)
        window_size = self.width  # Number of samples to display in one frame

        for i in range(self.num_frames):
            if self.scroll_direction == "left":
                start_idx = i * samples_per_frame
            elif self.scroll_direction == "right":
                start_idx = self.total_samples - window_size - i * samples_per_frame
            else:
                start_idx = i * samples_per_frame  # Default to left scroll

            end_idx = start_idx + window_size
            if end_idx > self.total_samples:
                end_idx = self.total_samples
                start_idx = max(0, end_idx - window_size)  # Adjust start_idx to maintain window size

            waveform_window = self.full_waveform[start_idx:end_idx]
            frame_surface = self._generate_waveform_frame(waveform_window)
            frame_array = self._surface_to_array(frame_surface)
            frames.append(frame_array)
            self.update_progress()
        self.end_progress()
        frames = np.stack(frames, axis=0)  # Shape: (B, H, W, C)
        frames = frames.astype(np.uint8)
        return torch.from_numpy(frames)

    def _generate_waveform_frame(self, waveform):
        self.screen.fill((0, 0, 0))  # Clear the screen with black background

        middle = self.height // 2
        x_scale = self.width / len(waveform)
        # Generate points for the waveform
        points = []
        for idx in range(len(waveform)):
            x = int(idx * x_scale)
            y = middle - int(waveform[idx] * middle)
            points.append((x, y))
        if len(points) > 1:
            # Draw the waveform on the surface
            pygame.draw.lines(self.screen, (255, 255, 255), False, points, 1)
        return self.screen.copy()

    def create_spectrogram(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Spectrogram with Pygame")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            frame_surface = self._generate_spectrogram_frame(audio_frame)
            frame_array = self._surface_to_array(frame_surface)
            frames.append(frame_array)
            self.update_progress()
        self.end_progress()
        frames = np.stack(frames, axis=0)  # Shape: (B, H, W, C)
        frames = frames.astype(np.uint8)
        return torch.from_numpy(frames)

    def _generate_spectrogram_frame(self, audio_frame):
        self.screen.fill((0, 0, 0))  # Clear the screen
        # Compute spectrogram
        n_fft = min(2048, len(audio_frame))
        hop_length = n_fft // 4
        S = np.abs(librosa.stft(audio_frame, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        S_db = (S_db + 80) / 80  # Normalize to [0,1]

        height, width = S_db.shape
        spectrogram_surface = pygame.Surface((width, height))
        for x in range(width):
            for y in range(height):
                intensity = int(S_db[y, x] * 255)
                spectrogram_surface.set_at((x, y), (intensity, intensity, intensity))

        # Scale spectrogram to desired size
        spectrogram_surface = pygame.transform.scale(spectrogram_surface, (self.width, self.height))
        self.screen.blit(spectrogram_surface, (0, 0))
        return self.screen.copy()

    def create_mfcc(self):
        frames = []
        self.start_progress(self.num_frames, "Creating MFCC with Pygame")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            frame_surface = self._generate_mfcc_frame(audio_frame)
            frame_array = self._surface_to_array(frame_surface)
            frames.append(frame_array)
            self.update_progress()
        self.end_progress()
        frames = np.stack(frames, axis=0)
        frames = frames.astype(np.uint8)
        return torch.from_numpy(frames)

    def _generate_mfcc_frame(self, audio_frame):
        self.screen.fill((0, 0, 0))  # Clear the screen
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio_frame, sr=self.sample_rate, n_mfcc=20)
        mfccs = (mfccs - np.min(mfccs)) / (np.max(mfccs) - np.min(mfccs))  # Normalize to [0,1]

        height, width = mfccs.shape
        mfcc_surface = pygame.Surface((width, height))
        for x in range(width):
            for y in range(height):
                intensity = int(mfccs[y, x] * 255)
                mfcc_surface.set_at((x, y), (intensity, intensity, intensity))

        # Scale MFCC to desired size
        mfcc_surface = pygame.transform.scale(mfcc_surface, (self.width, self.height))
        self.screen.blit(mfcc_surface, (0, 0))
        return self.screen.copy()

    def create_chroma(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Chroma with Pygame")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            frame_surface = self._generate_chroma_frame(audio_frame)
            frame_array = self._surface_to_array(frame_surface)
            frames.append(frame_array)
            self.update_progress()
        self.end_progress()
        frames = np.stack(frames, axis=0)
        frames = frames.astype(np.uint8)
        return torch.from_numpy(frames)

    def _generate_chroma_frame(self, audio_frame):
        self.screen.fill((0, 0, 0))  # Clear the screen
        # Compute Chroma features
        chromagram = librosa.feature.chroma_stft(y=audio_frame, sr=self.sample_rate)
        chromagram = (chromagram - np.min(chromagram)) / (np.max(chromagram) - np.min(chromagram))  # Normalize to [0,1]

        height, width = chromagram.shape
        chroma_surface = pygame.Surface((width, height))
        for x in range(width):
            for y in range(height):
                intensity = int(chromagram[y, x] * 255)
                chroma_surface.set_at((x, y), (intensity, intensity, intensity))

        # Scale Chroma to desired size
        chroma_surface = pygame.transform.scale(chroma_surface, (self.width, self.height))
        self.screen.blit(chroma_surface, (0, 0))
        return self.screen.copy()

    def create_tonnetz(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Tonnetz with Pygame")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            frame_surface = self._generate_tonnetz_frame(audio_frame)
            frame_array = self._surface_to_array(frame_surface)
            frames.append(frame_array)
            self.update_progress()
        self.end_progress()
        frames = np.stack(frames, axis=0)
        frames = frames.astype(np.uint8)
        return torch.from_numpy(frames)

    def _generate_tonnetz_frame(self, audio_frame):
        self.screen.fill((0, 0, 0))  # Clear the screen
        # Compute Tonnetz
        tonnetz = librosa.feature.tonnetz(y=audio_frame, sr=self.sample_rate)
        tonnetz = (tonnetz - np.min(tonnetz)) / (np.max(tonnetz) - np.min(tonnetz))  # Normalize to [0,1]

        height, width = tonnetz.shape
        tonnetz_surface = pygame.Surface((width, height))
        for x in range(width):
            for y in range(height):
                intensity = int(tonnetz[y, x] * 255)
                tonnetz_surface.set_at((x, y), (intensity, intensity, intensity))

        # Scale Tonnetz to desired size
        tonnetz_surface = pygame.transform.scale(tonnetz_surface, (self.width, self.height))
        self.screen.blit(tonnetz_surface, (0, 0))
        return self.screen.copy()

    def create_spectral_centroid(self):
        frames = []
        self.start_progress(self.num_frames, "Creating Spectral Centroid with Pygame")
        for i in range(self.num_frames):
            audio_frame = self._get_audio_frame(i)
            frame_surface = self._generate_spectral_centroid_frame(audio_frame)
            frame_array = self._surface_to_array(frame_surface)
            frames.append(frame_array)
            self.update_progress()
        self.end_progress()
        frames = np.stack(frames, axis=0)
        frames = frames.astype(np.uint8)
        return torch.from_numpy(frames)

    def _generate_spectral_centroid_frame(self, audio_frame):
        self.screen.fill((0, 0, 0))  # Clear the screen
        # Compute Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio_frame, sr=self.sample_rate)
        centroid = (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid))  # Normalize to [0,1]

        height, width = centroid.shape
        centroid_surface = pygame.Surface((width, height))
        for x in range(width):
            for y in range(height):
                intensity = int(centroid[y, x] * 255)
                centroid_surface.set_at((x, y), (intensity, intensity, intensity))

        # Scale Centroid to desired size
        centroid_surface = pygame.transform.scale(centroid_surface, (self.width, self.height))
        self.screen.blit(centroid_surface, (0, 0))
        return self.screen.copy()

    def __del__(self):
        pygame.quit()