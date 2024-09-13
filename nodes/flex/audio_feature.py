from . features import BaseFeature  
import librosa
import numpy as np
from scipy.signal import hilbert

class BaseAudioFeature(BaseFeature):
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
        self.available_features = [
            'amplitude_envelope',
            'rms_energy',
            'spectral_centroid',
            'onset_strength',
            'chroma_features'
        ]
        self.feature_name = feature_type
        self._prepare_audio()

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
        self.available_features = [
            'pitch',
            'pitch_filtered',
            'pitch_direction',
            'vibrato_signal',
            'vibrato_intensity',
        ]
        self.feature_type = feature_type
        self.pitch_range_collections = pitch_range_collections or []  # List of collections
        self.vibrato_options = vibrato_options  # Dict with vibrato analysis options
        self.feature_name = feature_type
        self.window_size = window_size  # Number of frames before and after
        self.pitch_tolerance = pitch_tolerance  # Tolerance for pitch variation
        self.current_frame = 0  # Initialize the current frame index
        self._prepare_audio()

    def extract(self):
        self.features = {self.feature_name: []}
        if self.feature_name == 'pitch':
            self._extract_pitch()
        elif self.feature_name == 'pitch_filtered':
            self._extract_pitch(filtered=True)
        elif self.feature_name == 'pitch_direction':
            self._extract_pitch_direction()
        elif self.feature_name == 'vibrato_signal':
            self._extract_vibrato(signal=True)
        elif self.feature_name == 'vibrato_intensity':
            self._extract_vibrato(intensity=True)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_name}")
        self._normalize_features()
        return self

    def _extract_pitch(self, filtered=False):
        pitches = []
        confidences = []
        prev_valid_pitch = None  # Keep track of the previous valid pitch
        for i in range(self.frame_count):
            self.current_frame = i  # Update current frame index
            window_frame = self._get_audio_window(i)
            pitch, confidence = self._calculate_pitch(window_frame)
            # Apply pitch tolerance
            if prev_valid_pitch is not None:
                if abs(pitch - prev_valid_pitch) < self.pitch_tolerance:
                    pitch = prev_valid_pitch
            # Apply pitch range filtering
            if filtered and self.pitch_range_collections:
                if self._matches_any_collection(pitch):
                    pitches.append(pitch)
                    prev_valid_pitch = pitch
                else:
                    pitches.append(0.0)
                    prev_valid_pitch = None
            else:
                pitches.append(pitch)
                prev_valid_pitch = pitch
            confidences.append(confidence)
        self.features[self.feature_name] = pitches

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
            pitch, _ = self._calculate_pitch(window_frame)
            if pitch > 0:
                pitches_in_window.append(pitch)
        # For each pitch range, check if there's any pitch in the window that matches
        return all(
            any(pr.contains(pitch) for pitch in pitches_in_window)
            for pr in pitch_ranges
        )

    def _extract_pitch_direction(self):
        pitches = []
        prev_valid_pitch = None
        for i in range(self.frame_count):
            self.current_frame = i  # Update current frame index
            window_frame = self._get_audio_window(i)
            pitch, _ = self._calculate_pitch(window_frame)
            # Apply pitch tolerance
            if prev_valid_pitch is not None:
                if abs(pitch - prev_valid_pitch) < self.pitch_tolerance:
                    pitch = prev_valid_pitch
            # Apply pitch range filtering
            if self.pitch_range_collections:
                if self._matches_any_collection(pitch):
                    pitches.append(pitch)
                    prev_valid_pitch = pitch
                else:
                    pitches.append(0.0)
                    prev_valid_pitch = None
            else:
                pitches.append(pitch)
                prev_valid_pitch = pitch
        # Calculate the difference between consecutive pitches
        pitches = np.array(pitches)
        pitch_diff = np.diff(pitches, prepend=pitches[0])
        # Apply pitch tolerance to pitch differences
        pitch_diff[np.abs(pitch_diff) < self.pitch_tolerance] = 0.0
        # Assign 1 for ascending, -1 for descending, 0 for no change
        direction = np.sign(pitch_diff)
        self.features[self.feature_name] = direction.tolist()

    def _extract_vibrato(self, signal=False, intensity=False):
        pitches = []
        prev_valid_pitch = None
        for i in range(self.frame_count):
            self.current_frame = i  # Update current frame index
            window_frame = self._get_audio_window(i)
            pitch, _ = self._calculate_pitch(window_frame)
            # Apply pitch tolerance
            if prev_valid_pitch is not None:
                if abs(pitch - prev_valid_pitch) < self.pitch_tolerance:
                    pitch = prev_valid_pitch
            # Apply pitch range filtering
            if self.pitch_range_collections:
                if self._matches_any_collection(pitch):
                    pitches.append(pitch)
                    prev_valid_pitch = pitch
                else:
                    pitches.append(0.0)
                    prev_valid_pitch = None
            else:
                pitches.append(pitch)
                prev_valid_pitch = pitch
        pitches = np.array(pitches)
        # Remove zeros for vibrato calculation
        valid_indices = pitches > 0
        valid_pitches = pitches[valid_indices]
        # Handle case where no valid pitches are found
        if len(valid_pitches) == 0:
            self.features[self.feature_name] = [0.0] * self.frame_count
            return
        # Estimate vibrato using Hilbert transform
        analytic_signal = hilbert(valid_pitches)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase)
        # Reconstruct full-length array
        feature_values = np.zeros_like(pitches)
        if signal:
            # Use instantaneous frequency as vibrato signal
            vibrato_signal = np.concatenate(([0], instantaneous_frequency))
            feature_values[valid_indices] = vibrato_signal
        elif intensity:
            # Use amplitude envelope as vibrato intensity
            feature_values[valid_indices] = amplitude_envelope
        else:
            raise ValueError("Specify either 'signal' or 'intensity' for vibrato extraction")
        self.features[self.feature_name] = feature_values.tolist()

    def _calculate_pitch(self, audio_frame):
        if audio_frame.size == 0:
            return 0.0, 0.0
        pitches, magnitudes = librosa.core.piptrack(y=audio_frame, sr=self.sample_rate)
        magnitudes = magnitudes.flatten()
        pitches = pitches.flatten()
        # Find the pitch with the highest magnitude
        if magnitudes.size == 0 or np.all(magnitudes == 0):
            return 0.0, 0.0
        index = magnitudes.argmax()
        pitch = pitches[index]
        confidence = magnitudes[index]
        # Handle NaN values
        pitch = pitch if not np.isnan(pitch) else 0.0
        confidence = confidence if not np.isnan(confidence) else 0.0
        return pitch, confidence

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
        self.features[self.feature_name] = normalized.tolist()

class PitchRange:
    def __init__(self, min_pitch, max_pitch):
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

    def contains(self, pitch):
        return self.min_pitch <= pitch <= self.max_pitch