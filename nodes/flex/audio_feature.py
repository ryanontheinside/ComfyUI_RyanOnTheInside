from . features import BaseFeature  
import librosa
import numpy as np

class BaseAudioFeature(BaseFeature):
    def __init__(self, name, audio, frame_rate, frame_count):
        super().__init__(name, "audio", frame_rate, frame_count)
        self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()
        self.sample_rate = audio['sample_rate']
        self.frame_duration = 1 / self.frame_rate if self.frame_rate > 0 else len(self.audio) / (self.sample_rate * self.frame_count)

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = (frame_index + 1) * self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        if start_sample >= len(self.audio):
            return np.array([])  # Return empty array if we've run out of audio
        return self.audio[start_sample:min(end_sample, len(self.audio))]

#TODO refactor to use BaseAudioFeature    
class AudioFeature(BaseFeature):
    def __init__(self, name, audio, num_frames, frame_rate, feature_type='amplitude_envelope'):
        self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()
        self.sample_rate = audio['sample_rate']
        self.num_frames = num_frames
        self.feature_type = feature_type
        super().__init__(name, "audio", frame_rate, num_frames)
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

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = (frame_index + 1) * self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        if start_sample >= len(self.audio):
            return np.array([])  # Return empty array if we've run out of audio
        return self.audio[start_sample:min(end_sample, len(self.audio))]

    def _amplitude_envelope(self):
        def safe_max(frame):
            return np.max(np.abs(frame)) if frame.size > 0 else 0
        return np.array([safe_max(self._get_audio_frame(i)) for i in range(self.num_frames)])

    def _rms_energy(self):
        return np.array([np.sqrt(np.mean(self._get_audio_frame(i)**2)) for i in range(self.num_frames)])

    def _spectral_centroid(self):
        return np.array([np.mean(librosa.feature.spectral_centroid(y=self._get_audio_frame(i), sr=self.sample_rate)[0]) for i in range(self.num_frames)])

    def _onset_detection(self):
        return np.array([np.mean(librosa.onset.onset_strength(y=self._get_audio_frame(i), sr=self.sample_rate)) for i in range(self.num_frames)])

    def _chroma_features(self):
        return np.array([np.mean(librosa.feature.chroma_stft(y=self._get_audio_frame(i), sr=self.sample_rate), axis=1) for i in range(self.num_frames)])

class PitchFeature(BaseFeature):
    def __init__(self, name, audio, num_frames, frame_rate, feature_type='fundamental_frequency', frame_duration=0.01):
        super().__init__(name, "pitch", frame_rate, num_frames)
        self.audio = np.nan_to_num(audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
        self.sample_rate = audio['sample_rate']
        self.feature_type = feature_type
        self.frame_duration = frame_duration
        self.data = None

    def extract(self):
        if self.feature_type == 'fundamental_frequency':
            self.data = self._extract_fundamental_frequency()
        elif self.feature_type == 'pitch_confidence':
            self.data = self._extract_pitch_confidence()
        elif self.feature_type == 'pitch_statistics':
            self.data = self._extract_pitch_statistics()
        elif self.feature_type == 'pitch_contour':
            self.data = self._extract_pitch_contour()
        elif self.feature_type == 'vibrato':
            self.data = self._extract_vibrato()
        elif self.feature_type == 'pitch_histogram':
            self.data = self._extract_pitch_histogram()
        elif self.feature_type == 'pitch_class_profile':
            self.data = self._extract_pitch_class_profile()
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
        
        return self.normalize()

    def _extract_fundamental_frequency(self):
        hop_length = int(self.frame_duration * self.sample_rate)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        return f0

    def _extract_pitch_confidence(self):
        hop_length = int(self.frame_duration * self.sample_rate)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            self.audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=hop_length
        )
        return voiced_probs

    def _extract_pitch_statistics(self):
        f0 = self._extract_fundamental_frequency()
        f0 = f0[f0 > 0]  # Remove unvoiced frames
        if len(f0) == 0:
            return np.zeros(self.frame_count)
        mean_pitch = np.mean(f0)
        pitch_range = np.max(f0) - np.min(f0)
        pitch_variance = np.var(f0)
        # Normalize and combine statistics
        normalized_stats = np.array([
            (mean_pitch - np.min(f0)) / (np.max(f0) - np.min(f0)),
            pitch_range / (np.max(f0) - np.min(f0)),
            pitch_variance / np.max(pitch_variance)
        ])
        # Remove NaN and Inf values
        normalized_stats = np.nan_to_num(normalized_stats, nan=0.0, posinf=1.0, neginf=-1.0)
        # Interpolate to match frame count
        return np.interp(
            np.linspace(0, 1, self.frame_count),
            np.linspace(0, 1, len(normalized_stats)),
            normalized_stats
        )

    def _extract_pitch_contour(self):
        f0 = self._extract_fundamental_frequency()
        # Calculate pitch slope (first derivative of pitch)
        pitch_slope = np.gradient(f0)
        # Remove NaN and Inf values
        pitch_slope = np.nan_to_num(pitch_slope, nan=0.0, posinf=1.0, neginf=-1.0)
        # Normalize slope
        normalized_slope = (pitch_slope - np.min(pitch_slope)) / (np.max(pitch_slope) - np.min(pitch_slope))
        # Interpolate to match frame count
        return np.interp(
            np.linspace(0, 1, self.frame_count),
            np.linspace(0, 1, len(normalized_slope)),
            normalized_slope
        )

    def _extract_vibrato(self):
        f0 = self._extract_fundamental_frequency()
        # Detrend the pitch
        detrended_f0 = f0 - librosa.feature.rms(y=f0, frame_length=len(f0), hop_length=1)[0]
        # Remove NaN and Inf values from detrended_f0
        detrended_f0 = np.nan_to_num(detrended_f0, nan=0.0, posinf=0.0, neginf=0.0)
        # Compute the spectrogram of the detrended pitch
        spec = np.abs(librosa.stft(detrended_f0))
        # Look for peaks in the 4-8 Hz range (typical vibrato rates)
        vibrato_range = np.logical_and(librosa.fft_frequencies(sr=self.sample_rate) >= 4,
                                       librosa.fft_frequencies(sr=self.sample_rate) <= 8)
        vibrato_strength = np.max(spec[vibrato_range, :], axis=0)
        # Normalize and interpolate
        normalized_vibrato = (vibrato_strength - np.min(vibrato_strength)) / (np.max(vibrato_strength) - np.min(vibrato_strength))
        # Remove NaN and Inf values
        normalized_vibrato = np.nan_to_num(normalized_vibrato, nan=0.0, posinf=1.0, neginf=-1.0)
        return np.interp(
            np.linspace(0, 1, self.frame_count),
            np.linspace(0, 1, len(normalized_vibrato)),
            normalized_vibrato
        )

    def _extract_pitch_histogram(self):
        f0 = self._extract_fundamental_frequency()
        f0 = f0[f0 > 0]  # Remove unvoiced frames
        if len(f0) == 0:
            return np.zeros(self.frame_count)
        hist, _ = np.histogram(f0, bins=12, range=(np.min(f0), np.max(f0)), density=True)
        # Normalize histogram
        normalized_hist = hist / np.max(hist)
        # Remove NaN and Inf values
        normalized_hist = np.nan_to_num(normalized_hist, nan=0.0, posinf=1.0, neginf=-1.0)
        # Interpolate to match frame count
        return np.interp(
            np.linspace(0, 1, self.frame_count),
            np.linspace(0, 1, len(normalized_hist)),
            normalized_hist
        )

    def _extract_pitch_class_profile(self):
        hop_length = int(self.frame_duration * self.sample_rate)
        chroma = librosa.feature.chroma_cqt(y=self.audio, sr=self.sample_rate, hop_length=hop_length)
        # Average across time
        mean_chroma = np.mean(chroma, axis=1)
        # Normalize
        normalized_chroma = mean_chroma / np.max(mean_chroma)
        # Remove NaN and Inf values
        normalized_chroma = np.nan_to_num(normalized_chroma, nan=0.0, posinf=1.0, neginf=-1.0)
        # Interpolate to match frame count
        return np.interp(
            np.linspace(0, 1, self.frame_count),
            np.linspace(0, 1, len(normalized_chroma)),
            normalized_chroma
        )

    def normalize(self):
        if self.data is not None:
            self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return self
