from .features import BaseFeature
import librosa
import numpy as np
from scipy.signal import hilbert
from scipy import signal
import functools
from abc  import ABC, abstractmethod
class BaseAudioFeature(BaseFeature):

    @classmethod
    @abstractmethod
    def get_extraction_methods(cls):
        """Return a list of parameter names that can be modulated."""
        return []
    
    def __init__(self, name, audio, frame_count, frame_rate):
        super().__init__(name, "audio", frame_rate, frame_count)
        self.audio = audio
        self.sample_rate = None
        self.frame_duration = None
        self.feature_name = None
        self.available_features = []

    def _prepare_audio(self):
        self.sample_rate = self.audio['sample_rate']
        waveform = self.audio['waveform']

        # Handle multi-dimensional tensors
        if waveform.ndim > 1:
            waveform = waveform.squeeze()

        # Handle multi-channel audio by averaging channels
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        self.audio_array = waveform.cpu().numpy()
        self.frame_duration = 1 / self.frame_rate

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = start_time + self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        if start_sample >= len(self.audio_array):
            return np.array([])  # Return empty array if we've run out of audio
        return self.audio_array[start_sample:min(end_sample, len(self.audio_array))]

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
            raise ValueError(
                f"Invalid feature name. Available features are: {', '.join(self.available_features)}"
            )
   
class AudioFeature(BaseAudioFeature):

    def __init__(self, feature_name, audio, frame_count, frame_rate, feature_type='amplitude_envelope'):
        super().__init__(feature_name, audio, frame_count, frame_rate)
        self.feature_type = feature_type
        self.available_features = self.get_extraction_methods()
        self.feature_name = feature_type
        self._prepare_audio()

    @classmethod
    def get_extraction_methods(cls):
        """Return a list of parameter names that can be modulated."""
        return [
            'amplitude_envelope',
            'rms_energy',
            'spectral_centroid',
            'onset_strength',
            'chroma_features'
        ]
    
    def extract(self):
        self.features = {self.feature_name: []}
        for i in range(self.frame_count):
            frame = self._get_audio_frame(i)
            value = self._calculate_feature(frame)
            self.features[self.feature_name].append(value)
        self._normalize_features()
        return self

    def _calculate_feature(self, frame):
        if frame.size == 0:
            return 0.0
        if self.feature_name == 'amplitude_envelope':
            return np.max(np.abs(frame))
        elif self.feature_name == 'rms_energy':
            return np.sqrt(np.mean(frame ** 2))
        elif self.feature_name == 'spectral_centroid':
            centroid = librosa.feature.spectral_centroid(y=frame, sr=self.sample_rate)
            return np.mean(centroid)
        elif self.feature_name == 'onset_strength':
            strength = librosa.onset.onset_strength(y=frame, sr=self.sample_rate)
            return np.mean(strength)
        elif self.feature_name == 'chroma_features':
            chroma = librosa.feature.chroma_stft(y=frame, sr=self.sample_rate)
            return np.mean(chroma)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_name}")

    def _normalize_features(self):
        feature_array = np.array(self.features[self.feature_name], dtype=np.float32)
        min_val = np.min(feature_array)
        max_val = np.max(feature_array)
        if max_val > min_val:
            normalized = (feature_array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(feature_array)
        self.features[self.feature_name] = normalized.tolist()

class PitchFeature(BaseAudioFeature):
    def __init__(
        self,
        feature_name,
        audio,
        frame_count,
        frame_rate,
        pitch_range_collections=None,
        feature_type='pitch',
        window_size=0,
        pitch_tolerance=0.0,
        vibrato_options=None,
    ):
        super().__init__(feature_name, audio, frame_count, frame_rate)
        self.available_features = self.get_extraction_methods()
        self.feature_type = feature_type
        self.pitch_range_collections = pitch_range_collections or []
        self.vibrato_options = vibrato_options
        self.feature_name = feature_type
        self.window_size = window_size
        self.pitch_tolerance = pitch_tolerance
        self.current_frame = 0
        self._prepare_audio()
        
        # Parameters for pitch detection
        self.frame_length = 2048
        self.hop_length = 512
        self.fmin = 20  # Lower frequency limit
        self.fmax = 4000  # Upper frequency limit

    @classmethod
    def get_extraction_methods(cls):
        """Return a list of parameter names that can be modulated."""
        return [
            'pitch',
            'pitch_direction',
            'pitch_contour',  # Added new feature
            'vibrato_signal',
            'vibrato_intensity',
        ]
    
    @classmethod
    def pitch_to_note(self, pitch):
        if pitch == 0:
            return "N/A"
        return librosa.hz_to_note(pitch)

    def ensure_pitch_and_confidence(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.feature_name + '_pitch' not in self.features:
                self._calculate_pitch_sequence()
            return func(self, *args, **kwargs)
        return wrapper

    def _calculate_pitch_sequence(self):
        pitches = []
        confidences = []
        
        # Use multiple pitch estimation methods
        pitch_piptrack, mag = librosa.piptrack(y=self.audio_array, sr=self.sample_rate, 
                                               fmin=self.fmin, fmax=self.fmax,
                                               hop_length=self.hop_length)
        
        pitch_yin = librosa.yin(self.audio_array, fmin=self.fmin, fmax=self.fmax, 
                                sr=self.sample_rate, hop_length=self.hop_length)
        
        for i in range(pitch_piptrack.shape[1]):
            pitch_candidates = pitch_piptrack[:, i]
            mag_candidates = mag[:, i]
            
            # Get the top 3 pitch candidates
            top_indices = np.argsort(mag_candidates)[-3:]
            top_pitches = pitch_candidates[top_indices]
            top_mags = mag_candidates[top_indices]
            
            # Compare with YIN pitch
            yin_pitch = pitch_yin[i]
            
            # Choose the pitch closest to YIN if it's within a threshold
            threshold = 50  # Hz
            closest_pitch = min(top_pitches, key=lambda x: abs(x - yin_pitch))
            
            if abs(closest_pitch - yin_pitch) < threshold:
                pitch = closest_pitch
                confidence = np.max(top_mags) / np.sum(top_mags)
            else:
                pitch = 0.0
                confidence = 0.0
            
            pitches.append(pitch)
            confidences.append(confidence)
        
        # Interpolate to match frame count
        frame_times = np.linspace(0, len(self.audio_array) / self.sample_rate, num=self.frame_count)
        pitch_times = librosa.frames_to_time(np.arange(len(pitches)), sr=self.sample_rate, hop_length=self.hop_length)
        
        interpolated_pitch = np.interp(frame_times, pitch_times, pitches)
        interpolated_confidence = np.interp(frame_times, pitch_times, confidences)
        
        self.features[self.feature_name + '_pitch'] = interpolated_pitch.tolist()
        self.features[self.feature_name + '_confidence'] = interpolated_confidence.tolist()
    
    def extract(self):
        self.features = {self.feature_name: []}
        if self.feature_name == 'pitch':
            self._extract_pitch_sequence(filtered=True)
        elif self.feature_name == 'pitch_direction':
            self._extract_pitch_direction()
        elif self.feature_name == 'pitch_contour':  # New condition
            self._extract_pitch_contour()
        elif self.feature_name == 'vibrato_signal':
            self._extract_vibrato(signal=True)
        elif self.feature_name == 'vibrato_intensity':
            self._extract_vibrato(intensity=True)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_name}")
        self._normalize_features()
        return self

    @ensure_pitch_and_confidence
    def _extract_pitch_sequence(self, filtered=False):
        pitches = np.array(self.features[self.feature_name + '_pitch'])
        confidences = np.array(self.features[self.feature_name + '_confidence'])
        
        # Apply filtering and smoothing
        filtered_pitch = self._filter_and_smooth_pitch(pitches, confidences, filtered)
        
        # Store the filtered pitch directly, as it's already a list
        self.features[self.feature_name] = filtered_pitch

    @ensure_pitch_and_confidence
    def _extract_pitch_contour(self):
        pitches = self.features[self.feature_name + '_pitch']

        valid_pitches = [p for p in pitches if p > 0 and np.isfinite(p)]

        if not valid_pitches:
            self.features[self.feature_name] = [0.0] * self.frame_count
            return

        min_pitch = min(valid_pitches)
        max_pitch = max(valid_pitches)
        pitch_range = max_pitch - min_pitch

        if pitch_range == 0:
            self.features[self.feature_name] = [0.0 if p == 0 else 1.0 for p in pitches]
        else:
            self.features[self.feature_name] = [
                0.0 if p == 0 else (p - min_pitch) / pitch_range for p in pitches
            ]
            
    def _matches_any_collection(self, pitch):
        for collection in self.pitch_range_collections:
            pitch_ranges = collection["pitch_ranges"]
            chord_only = collection.get("chord_only", False)
            if chord_only:
                # Check if all pitch ranges in the collection are present
                if self._all_pitches_present(collection):
                    return True
            else:
                # Check if the pitch matches any pitch range in the collection
                if any(pr.contains(pitch) for pr in pitch_ranges):
                    return True
        return False

    def _all_pitches_present(self, collection):
        pitch_ranges = collection["pitch_ranges"]
        chord_window_size = collection.get("window_size", 0)  # You may define this per collection
        # Calculate the start and end frames for the window
        start_frame = max(0, self.current_frame - chord_window_size)
        end_frame = min(self.frame_count - 1, self.current_frame + chord_window_size)
        pitches_in_window = []
        for i in range(start_frame, end_frame + 1):
            window_frame = self._get_audio_window(i)
            pitch, _ = self._calculate_robust_frame_pitch(window_frame)
            if pitch > 0:
                pitches_in_window.append(pitch)
        # For each pitch range, check if there's any pitch in the window that matches
        return all(
            any(pr.contains(pitch) for pitch in pitches_in_window)
            for pr in pitch_ranges
        )

    @ensure_pitch_and_confidence
    def _extract_pitch_direction(self):
        pitches = self.features[self.feature_name + '_pitch']
        
        pitch_diff = np.diff(pitches, prepend=pitches[0])
        pitch_diff[np.abs(pitch_diff) < self.pitch_tolerance] = 0.0
        direction = np.sign(pitch_diff)
        self.features[self.feature_name] = direction.tolist()

    @ensure_pitch_and_confidence
    def _extract_vibrato(self, signal=False, intensity=False):
        pitches = np.array(self.features[self.feature_name + '_pitch'])
        
        valid_indices = pitches > 0
        valid_pitches = pitches[valid_indices]
        
        if len(valid_pitches) == 0:
            self.features[self.feature_name] = [0.0] * self.frame_count
            return
        
        analytic_signal = hilbert(valid_pitches)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase)
        
        feature_values = np.zeros_like(pitches)
        if signal:
            vibrato_signal = np.concatenate(([0], instantaneous_frequency))
            feature_values[valid_indices] = vibrato_signal
        elif intensity:
            feature_values[valid_indices] = amplitude_envelope
        else:
            raise ValueError("Specify either 'signal' or 'intensity' for vibrato extraction")
        
        self.features[self.feature_name] = feature_values.tolist()

    def _get_audio_window(self, frame_index):
        # Calculate the start and end frame indices for the window
        start_frame = max(0, frame_index - self.window_size)
        end_frame = min(self.frame_count - 1, frame_index + self.window_size)
        # Calculate the corresponding audio sample positions
        start_time = start_frame * self.frame_duration
        end_time = (end_frame + 1) * self.frame_duration  # +1 to include the end frame
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        if start_sample >= len(self.audio_array):
            return np.array([])  # Return empty array if we've run out of audio
        return self.audio_array[start_sample:min(end_sample, len(self.audio_array))]

    def _normalize_features(self):
        feature_array = np.array(self.features[self.feature_name], dtype=np.float32)
        finite_mask = np.isfinite(feature_array)
        if not np.any(finite_mask):
            normalized = np.zeros_like(feature_array)
        else:
            min_val = np.min(feature_array[finite_mask])
            max_val = np.max(feature_array[finite_mask])
            if max_val > min_val:
                normalized = np.zeros_like(feature_array)
                normalized[finite_mask] = (feature_array[finite_mask] - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(feature_array)
        self.features[self.feature_name+'_original'] =  self.features[self.feature_name].copy()
        self.features[self.feature_name] = normalized.tolist()

    def get_pitch_at_frame(self, frame_index):
        if self.data is not None:
            return self.data[frame_index]
        elif self.features is not None and hasattr(self, 'feature_name'):
            pitch = self.features[self.feature_name+'_pitch'][frame_index]
            confidence_key = self.feature_name+'_confidence'
            confidence = self.features.get(confidence_key, [1.0] * len(self.features[self.feature_name]))[frame_index]
            return (pitch, confidence)
        else:
            raise ValueError("No data or features available")

    def _filter_and_smooth_pitch(self, pitches, confidences, filtered):
        smoothed_pitch = signal.medfilt(pitches, kernel_size=5)
        
        prev_valid_pitch = None
        filtered_pitch = []
        
        for i, pitch in enumerate(smoothed_pitch):
            confidence = confidences[i]
            
            if confidence < 0.3:  # Adjust this threshold as needed
                pitch = 0.0
            
            if prev_valid_pitch is not None and abs(pitch - prev_valid_pitch) < self.pitch_tolerance:
                pitch = prev_valid_pitch
            
            if filtered and self.pitch_range_collections:
                if self._matches_any_collection(pitch):
                    filtered_pitch.append(pitch)
                    prev_valid_pitch = pitch
                else:
                    filtered_pitch.append(0.0)
                    prev_valid_pitch = None
            else:
                filtered_pitch.append(pitch)
                prev_valid_pitch = pitch
        
        return filtered_pitch
        
class PitchRange:
    def __init__(self, min_pitch, max_pitch):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    def contains(self, pitch):
        return self.min_pitch <= pitch <= self.max_pitch