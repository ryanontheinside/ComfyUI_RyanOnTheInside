import torch
import numpy as np
import cv2
from abc import abstractmethod
from comfy.utils import ProgressBar
from ..flex.flex_base import FlexBase
from ... import RyanOnTheInside
from ...tooltips import apply_tooltips

import matplotlib.pyplot as plt

class BaseAudioProcessor:
    def __init__(self, audio, num_frames, height, width, frame_rate):
        """
        Base class to process audio data.

        Parameters:
        - audio: dict with 'waveform' and 'sample_rate'.
        - num_frames: int, total number of frames to process.
        - height: int, height of the output image.
        - width: int, width of the output image.
        - frame_rate: float, frame rate for processing.
        """
        # Convert waveform tensor to mono numpy array
        self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()
        self.sample_rate = audio['sample_rate']
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_rate = frame_rate

        self.audio_duration = len(self.audio) / self.sample_rate
        self.frame_duration = 1 / self.frame_rate if self.frame_rate > 0 else self.audio_duration / self.num_frames

        self.spectrum = None  # Initialize spectrum

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

    def compute_spectrum(self, frame_index, fft_size, min_frequency, max_frequency):
        audio_frame = self._get_audio_frame(frame_index)
        if len(audio_frame) < fft_size:
            audio_frame = np.pad(audio_frame, (0, fft_size - len(audio_frame)), mode='constant')

        # Apply window function
        window = np.hanning(len(audio_frame))
        audio_frame = audio_frame * window

        # Compute FFT
        spectrum = np.abs(np.fft.rfft(audio_frame, n=fft_size))

        # Extract desired frequency range
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        freq_indices = np.where((freqs >= min_frequency) & (freqs <= max_frequency))[0]
        spectrum = spectrum[freq_indices]

        # Check if spectrum is not empty
        if spectrum.size > 0:
            # Apply logarithmic scaling
            spectrum = np.log1p(spectrum)

            # Normalize
            max_spectrum = np.max(spectrum)
            if max_spectrum != 0:
                spectrum = spectrum / max_spectrum
            else:
                spectrum = np.zeros_like(spectrum)
        else:
            # Return zeros if spectrum is empty
            spectrum = np.zeros(1)

        return spectrum

    def update_spectrum(self, new_spectrum, smoothing):
        if self.spectrum is None or len(self.spectrum) != len(new_spectrum):
            self.spectrum = np.zeros(len(new_spectrum))

        # Apply smoothing
        self.spectrum = smoothing * self.spectrum + (1 - smoothing) * new_spectrum

@apply_tooltips
class FlexAudioVisualizerBase(FlexBase, RyanOnTheInside):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})

        new_inputs = {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "screen_width": ("INT", {"default": 768, "min": 100, "max": 1920, "step": 1}),
                "screen_height": ("INT", {"default": 464, "min": 100, "max": 1080, "step": 1}),
                "audio_feature_param": (cls.get_modifiable_params(), {"default": "None"}),
                "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "opt_feature": ("FEATURE",),
                "opt_feature_pipe": ("FEATURE_PIPE",),
            }
        }

        required = {**base_required, **new_inputs["required"]}
        optional = {**base_optional, **new_inputs["optional"]}

        return {
            "required": required,
            "optional": optional
        }

    CATEGORY = "RyanOnTheInside/FlexAudioVisualizer"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "apply_effect"

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        pass

    def validate_param(self, param_name, param_value):
        """
        Ensure that modulated parameter values stay within valid ranges.
        """
        valid_params = {
            'fft_size': lambda x: max(256, int(2 ** np.round(np.log2(x)))) if x > 0 else 256,
            'min_frequency': lambda x: max(20.0, min(x, 20000.0)),
            'max_frequency': lambda x: max(20.0, min(x, 20000.0)),
            'num_bars': lambda x: max(1, int(x)),
            'num_points': lambda x: max(3, int(x)),
            'smoothing': lambda x: np.clip(x, 0.0, 1.0),
            'rotation': lambda x: x % 360.0,
            'curvature': lambda x: max(0.0, x),
            'separation': lambda x: max(0.0, x),
            'max_height': lambda x: max(10.0, x),
            'min_height': lambda x: max(0.0, x),
            'position_x': lambda x: np.clip(x, 0.0, 1.0),
            'position_y': lambda x: np.clip(x, 0.0, 1.0),
            'reflect': lambda x: bool(x),
            'line_width': lambda x: max(1, int(x)),
            'radius': lambda x: max(1.0, x),
            'base_radius': lambda x: max(1.0, x),
            'amplitude_scale': lambda x: max(0.0, x),
            # Add other parameters as needed
        }

        if param_name in valid_params:
            return valid_params[param_name](param_value)
        else:
            return param_value

    def rotate_image(self, image, angle):
        """Rotate the image by the given angle."""
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))

        return rotated_image

    @abstractmethod
    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        """
        Abstract method to get audio data for visualization at a specific frame index.
        Should process the audio data required for visualization.
        """
        pass

    @abstractmethod
    def draw(self, processor: BaseAudioProcessor, **kwargs) -> np.ndarray:
        """
        Abstract method to generate the image for the current frame.
        Returns:
        - image: numpy array of shape (H, W, 3).
        """
        pass

    def process_audio_data(self, processor: BaseAudioProcessor, frame_index, visualization_feature, num_points, smoothing, fft_size, min_frequency, max_frequency):
        # Keep the entire process_audio_data method from HEAD
        # This handles the core audio processing logic
        ...

    def apply_effect(self, *args, **kwargs):
        return self.apply_effect_internal(*args, **kwargs)

    def apply_effect_internal(self, audio, frame_rate, screen_width, screen_height, strength, feature_param, feature_mode,
                         audio_feature_param, opt_feature=None, **kwargs):
        # Keep the apply_effect implementation from base-class-refactor
        # But modify the get_audio_data call to use process_audio_data
        ...
        
        for i in range(num_frames):
            frame_kwargs = kwargs.copy()
            
            # Get audio data using the process_audio_data method
            spectrum, audio_feature_value = self.process_audio_data(
                processor, 
                i,
                frame_kwargs.get('visualization_feature'),
                frame_kwargs.get('num_points', frame_kwargs.get('num_bars')),
                frame_kwargs.get('smoothing'),
                frame_kwargs.get('fft_size'),
                frame_kwargs.get('min_frequency'),
                frame_kwargs.get('max_frequency')
            )
            
            # Continue with the rest of apply_effect_internal
            ...

@apply_tooltips
class FlexAudioVisualizerLine(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})

        new_inputs = {
            "required": {
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                # Parameters common to both methods/features
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                # Additional parameters
                "num_bars": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "max_height": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 2000.0, "step": 10.0}),
                "min_height": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 500.0, "step": 5.0}),
                "separation": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "curvature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "reflect": ("BOOLEAN", {"default": False}),
                "curve_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
            }
        }

        required = {**base_required, **new_inputs["required"]}
        optional = base_optional

        return {
            "required": required,
            "optional": optional
        }

    FUNCTION = "apply_effect"

    @classmethod
    def get_modifiable_params(cls):
        return ["smoothing", "rotation", "position_y",
                "num_bars", "max_height", "min_height", "separation", "curvature", "reflect",
                "curve_smoothing", "fft_size", "min_frequency", "max_frequency", "None"]

    def __init__(self):
        super().__init__()
        self.bars = None

    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        visualization_feature = kwargs.get('visualization_feature')
        smoothing = kwargs.get('smoothing')
        num_bars = kwargs.get('num_bars')

        fft_size = kwargs.get('fft_size')
        min_frequency = kwargs.get('min_frequency')
        max_frequency = kwargs.get('max_frequency')

        # Use the base class method
        self.bars, feature_value = self.process_audio_data(
            processor,
            frame_index,
            visualization_feature,
            num_bars,
            smoothing,
            fft_size,
            min_frequency,
            max_frequency
        )

        return feature_value

    def draw(self, processor: BaseAudioProcessor, **kwargs):
        visualization_method = kwargs.get('visualization_method')
        screen_width = processor.width
        screen_height = processor.height
        rotation = kwargs.get('rotation') % 360
        position_y = kwargs.get('position_y')
        reflect = kwargs.get('reflect')
        num_bars = kwargs.get('num_bars')  # Get the potentially modulated num_bars

        image = np.zeros((screen_height, screen_width, 3), dtype=np.float32)

        # Ensure self.bars matches the current num_bars
        if self.bars is None or len(self.bars) != num_bars:
            if self.bars is not None:
                # Interpolate existing data to match new num_bars
                old_indices = np.linspace(0, 1, len(self.bars))
                new_indices = np.linspace(0, 1, num_bars)
                self.bars = np.interp(new_indices, old_indices, self.bars)
            else:
                self.bars = np.zeros(num_bars)

        max_height = kwargs.get('max_height')
        min_height = kwargs.get('min_height')

        if visualization_method == 'bar':
            curvature = kwargs.get('curvature')
            separation = kwargs.get('separation')


            # Calculate bar width
            total_separation = separation * (num_bars - 1)
            total_bar_width = screen_width - total_separation
            bar_width = total_bar_width / num_bars

            # Baseline Y position
            baseline_y = int(screen_height * position_y)

            # Draw bars
            for i, bar_value in enumerate(self.bars):
                x = int(i * (bar_width + separation))

                bar_h = min_height + (max_height - min_height) * bar_value

                # Draw bar depending on reflect
                if reflect:
                    y_start = int(baseline_y - bar_h)
                    y_end = int(baseline_y + bar_h)
                else:
                    y_start = int(baseline_y - bar_h)
                    y_end = int(baseline_y)

                # Ensure y_start and y_end are within bounds
                y_start = max(min(y_start, screen_height - 1), 0)
                y_end = max(min(y_end, screen_height - 1), 0)

                # Swap y_start and y_end if necessary
                if y_start > y_end:
                    y_start, y_end = y_end, y_start

                x_end = int(x + bar_width)
                color = (1.0, 1.0, 1.0)  # White color

                # Draw rectangle with optional curvature
                rect_width = max(1, int(bar_width))
                rect_height = max(1, int(y_end - y_start))

                if curvature > 0 and rect_width > 1 and rect_height > 1:
                    rect = np.zeros((rect_height, rect_width, 3), dtype=np.float32)
                    # Create mask for rounded rectangle
                    radius = max(1, min(int(curvature), rect_width // 2, rect_height // 2))
                    mask = np.full((rect_height, rect_width), 0, dtype=np.uint8)
                    cv2.rectangle(mask, (0, 0), (rect_width - 1, rect_height - 1), 255, -1)
                    if radius > 1:
                        mask = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)
                    # Apply mask
                    rect[mask > 0] = color
                    # Place rect onto image
                    image[y_start:y_end, x:x + rect_width] = rect
                else:
                    cv2.rectangle(image, (x, y_start), (x_end, y_end), color, thickness=-1)
        elif visualization_method == 'line':
            curve_smoothing = kwargs.get('curve_smoothing')


            # Baseline Y position
            baseline_y = screen_height * position_y

            # Apply curve smoothing if specified
            data = self.bars
            if curve_smoothing > 0:
                window_size = int(len(data) * curve_smoothing)
                if window_size % 2 == 0:
                    window_size += 1  # Make it odd
                if window_size > 2:
                    data_smooth = self.smooth_curve(data, window_size)
                else:
                    data_smooth = data
            else:
                data_smooth = data

            # Compute amplitude
            amplitude_range = max_height - min_height
            amplitude = min_height + data_smooth * amplitude_range

            # X-axis
            num_points = len(amplitude)
            x_values = np.linspace(0, screen_width, num_points)

            if reflect:
                # Reflect the visualization
                y_values_up = baseline_y - amplitude
                y_values_down = baseline_y + amplitude

                points_up = np.array([x_values, y_values_up]).T.astype(np.int32)
                points_down = np.array([x_values, y_values_down]).T.astype(np.int32)

                # Draw the curves
                if len(points_up) > 1:
                    cv2.polylines(image, [points_up], False, (1.0, 1.0, 1.0))
                if len(points_down) > 1:
                    cv2.polylines(image, [points_down], False, (1.0, 1.0, 1.0))
            else:
                # Single visualization
                y_values = baseline_y - amplitude

                points = np.array([x_values, y_values]).T.astype(np.int32)

                if len(points) > 1:
                    cv2.polylines(image, [points], False, (1.0, 1.0, 1.0))

        # Apply rotation if needed
        if rotation != 0:
            image = self.rotate_image(image, rotation)

        return image

    def smooth_curve(self, y, window_size):
        """Apply a moving average to smooth the curve."""
        if window_size < 3:
            return y  # No smoothing needed
        box = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

@apply_tooltips
class FlexAudioVisualizerCircular(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})

        new_inputs = {
            "required": {
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                # Parameters common to both methods/features
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "num_points": ("INT", {"default": 360, "min": 3, "max": 1000, "step": 1}),
                # Additional parameters
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "radius": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "amplitude_scale": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 10.0}),
                "base_radius": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
            }
        }

        required = {**base_required, **new_inputs["required"]}
        optional = base_optional

        return {
            "required": required,
            "optional": optional
        }

    FUNCTION = "apply_effect"

    @classmethod
    def get_modifiable_params(cls):
        return ["smoothing", "rotation", "num_points",
                "fft_size", "min_frequency", "max_frequency", "radius", "line_width",
                "amplitude_scale", "base_radius", "None"]

    def __init__(self):
        super().__init__()

    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        visualization_feature = kwargs.get('visualization_feature')
        smoothing = kwargs.get('smoothing')
        num_points = kwargs.get('num_points')

        fft_size = kwargs.get('fft_size')
        min_frequency = kwargs.get('min_frequency')
        max_frequency = kwargs.get('max_frequency')

        # Use the base class method
        processor.spectrum, feature_value = self.process_audio_data(
            processor,
            frame_index,
            visualization_feature,
            num_points,
            smoothing,
            fft_size,
            min_frequency,
            max_frequency
        )

        return feature_value

    def draw(self, processor: BaseAudioProcessor, **kwargs):
        visualization_method = kwargs.get('visualization_method')
        rotation = kwargs.get('rotation') % 360
        num_points = kwargs.get('num_points')
        screen_width = processor.width
        screen_height = processor.height
        position_x = kwargs.get('position_x')
        position_y = kwargs.get('position_y')
        radius = kwargs.get('radius')
        base_radius = kwargs.get('base_radius')
        amplitude_scale = kwargs.get('amplitude_scale')
        line_width = kwargs.get('line_width')
        image = np.zeros((screen_height, screen_width, 3), dtype=np.float32)

        # Center based on position parameters
        center_x = screen_width * position_x
        center_y = screen_height * position_y

        # Angles for each point (with rotation)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        rotation_rad = np.deg2rad(rotation)
        angles += rotation_rad

        data = processor.spectrum

        if visualization_method == 'bar':
            # Compute end points of lines based on data
            for angle, amplitude in zip(angles, data):
                # Start point (on the base radius)
                x_start = center_x + base_radius * np.cos(angle)
                y_start = center_y + base_radius * np.sin(angle)

                # End point (from base_radius, extending by amplitude_scale)
                x_end = center_x + (base_radius + amplitude * amplitude_scale) * np.cos(angle)
                y_end = center_y + (base_radius + amplitude * amplitude_scale) * np.sin(angle)

                # Draw line
                cv2.line(
                    image,
                    (int(x_start), int(y_start)),
                    (int(x_end), int(y_end)),
                    (1.0, 1.0, 1.0),
                    thickness=line_width,
                )
        elif visualization_method == 'line':
            # Compute points for the deformed circle
            radii = base_radius + data * amplitude_scale
            
            # Compute x and y coordinates
            x_values = center_x + radii * np.cos(angles)
            y_values = center_y + radii * np.sin(angles)

            # Create points list
            points = np.array([x_values, y_values]).T.astype(np.int32)

            # Draw the deformed circle
            if len(points) > 2:
                cv2.polylines(image, [points], isClosed=True, color=(1.0, 1.0, 1.0), thickness=line_width)

        return image
