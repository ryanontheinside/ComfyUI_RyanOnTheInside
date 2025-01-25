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
                "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "opt_feature": ("FEATURE",),
            }
        }

        required = {**base_required, **new_inputs["required"]}
        optional = {**base_optional, **new_inputs["optional"]}

        return {
            "required": required,
            "optional": optional
        }

    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/Audio/Visualizers"
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
    def apply_effect_internal(self, processor: BaseAudioProcessor, **kwargs) -> np.ndarray:
        """
        Abstract method to generate the image for the current frame.
        Returns:
        - image: numpy array of shape (H, W, 3).
        """
        pass

    def process_audio_data(self, processor: BaseAudioProcessor, frame_index, visualization_feature, num_points, smoothing, fft_size, min_frequency, max_frequency):
        if visualization_feature == 'frequency':
            spectrum = processor.compute_spectrum(frame_index, fft_size, min_frequency, max_frequency)

            # Resample the spectrum to match the number of points
            data = np.interp(
                np.linspace(0, len(spectrum), num_points, endpoint=False),
                np.arange(len(spectrum)),
                spectrum,
            )

        elif visualization_feature == 'waveform':
            audio_frame = processor._get_audio_frame(frame_index)
            if len(audio_frame) < 1:
                data = np.zeros(num_points)
            else:
                # Use the waveform data directly
                data = np.interp(
                    np.linspace(0, len(audio_frame), num_points, endpoint=False),
                    np.arange(len(audio_frame)),
                    audio_frame,
                )
                # Normalize the waveform to [-1, 1]
                max_abs_value = np.max(np.abs(data))
                if max_abs_value != 0:
                    data = data / max_abs_value
                else:
                    data = np.zeros_like(data)
        else:
            data = np.zeros(num_points)

        # Update processor's spectrum with smoothing
        if processor.spectrum is None or len(processor.spectrum) != len(data):
            processor.spectrum = np.zeros(len(data))
        processor.update_spectrum(data, smoothing)

        # Return updated data and feature value
        feature_value = np.mean(np.abs(processor.spectrum))
        return processor.spectrum.copy(), feature_value

    def apply_effect(self, audio, frame_rate, screen_width, screen_height, 
                     strength, feature_param, feature_mode, feature_threshold,
                    opt_feature=None, **kwargs):
        # Calculate num_frames based on audio duration and frame_rate
        audio_duration = len(audio['waveform'].squeeze(0).mean(axis=0)) / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)

        # Initialize the audio processor
        processor = BaseAudioProcessor(audio, num_frames, screen_height, screen_width, frame_rate)

        # Initialize results list
        result = []

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        for i in range(num_frames):
            processor.current_frame = i
            
            # First process parameters to get the correct values for this frame
            processed_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=self.get_feature_value(i, opt_feature) if opt_feature is not None else None,
                feature_param=feature_param if opt_feature is not None else None,
                feature_mode=feature_mode if opt_feature is not None else None,
                strength=strength,
                feature_threshold=feature_threshold,
                **kwargs
            )
            processed_kwargs["frame_index"] = i
            
            # Get audio data using the processed parameters
            spectrum, _ = self.process_audio_data(
                processor, 
                i,
                processed_kwargs.get('visualization_feature'),
                processed_kwargs.get('num_points', processed_kwargs.get('num_bars')),
                processed_kwargs.get('smoothing'),
                processed_kwargs.get('fft_size'),
                processed_kwargs.get('min_frequency'),
                processed_kwargs.get('max_frequency')
            )

            # Generate the image for the current frame
            image = self.apply_effect_internal(processor, **processed_kwargs)
            result.append(image)

            self.update_progress()

        self.end_progress()

        # Convert result to tensor
        result_np = np.stack(result)
        result_tensor = torch.from_numpy(result_np).float()
        mask = result_tensor[:, :, :, 0]
        
        return (result_tensor, mask,)

@apply_tooltips
class FlexAudioVisualizerLine(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        base_required["feature_param"] = cls.get_modifiable_params()
        new_inputs = {
            "required": {
                "visualization_method": (["bar", "line"], {"default": "bar"}),

                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                # Parameters common to both methods/features
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4000.0, "step": 10.0}),
                # Parameters common to both methods/features
                "num_bars": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "max_height": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 2000.0, "step": 10.0}),
                "min_height": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 500.0, "step": 5.0}),
                "separation": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "curvature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "curve_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "reflect": ("BOOLEAN", {"default": False}),
            }
        }

        required = {**base_required, **new_inputs["required"]}
        optional = base_optional

        return {
            "required": required,
            "optional": optional
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["smoothing", "rotation", "position_y",
                "num_bars", "max_height", "min_height", "separation", "curvature", "reflect",
                "curve_smoothing", "fft_size", "min_frequency", "max_frequency", "None"]

    def __init__(self):
        super().__init__()

    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        visualization_feature = kwargs.get('visualization_feature')
        smoothing = kwargs.get('smoothing')
        num_bars = kwargs.get('num_bars')

        fft_size = kwargs.get('fft_size')
        min_frequency = kwargs.get('min_frequency')
        max_frequency = kwargs.get('max_frequency')

        # Use the base class method
        _, feature_value = self.process_audio_data(
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

    def apply_effect_internal(self, processor: BaseAudioProcessor, **kwargs):
        visualization_method = kwargs.get('visualization_method')
        screen_width = processor.width
        screen_height = processor.height
        rotation = kwargs.get('rotation') % 360
        position_y = kwargs.get('position_y')
        position_x = kwargs.get('position_x')
        reflect = kwargs.get('reflect', False)
        num_bars = kwargs.get('num_bars')
        length = kwargs.get('length', 0.0)
        max_height = kwargs.get('max_height')
        min_height = kwargs.get('min_height')

        # Calculate visualization length based on rotation when length is 0
        if length == 0:
            # Convert rotation to radians
            rotation_rad = np.deg2rad(rotation)
            # For rotation 0° or 180°, use width
            # For rotation 90° or 270°, use height
            # For other angles, calculate the appropriate length
            cos_theta = abs(np.cos(rotation_rad))
            sin_theta = abs(np.sin(rotation_rad))
            
            if cos_theta > sin_theta:
                # Closer to horizontal (0° or 180°)
                visualization_length = screen_width / cos_theta
            else:
                # Closer to vertical (90° or 270°)
                visualization_length = screen_height / sin_theta
        else:
            visualization_length = length

        # Create a larger canvas to handle rotation without clipping
        # Ensure all dimensions are integers
        padding = int(max(visualization_length, max_height) * 0.5)
        padded_width = int(visualization_length + 2 * padding)
        padded_height = int(screen_height + 2 * padding)
        padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.float32)

        # Get the current spectrum data
        data = processor.spectrum

        if visualization_method == 'bar':
            curvature = kwargs.get('curvature')
            separation = kwargs.get('separation')

            # Calculate bar width based on visualization length
            total_separation = separation * (num_bars - 1)
            total_bar_width = visualization_length - total_separation
            bar_width = total_bar_width / num_bars

            # Center the visualization at the middle of the padded image
            baseline_y = padded_height // 2
            x_offset = (padded_width - visualization_length) // 2

            # Draw bars
            for i, bar_value in enumerate(data):
                x = int(x_offset + i * (bar_width + separation))

                bar_h = min_height + (max_height - min_height) * bar_value

                # Draw bar based on reflect direction
                if reflect:
                    y_start = int(baseline_y)
                    y_end = int(baseline_y + bar_h)
                else:
                    y_start = int(baseline_y - bar_h)
                    y_end = int(baseline_y)

                # Ensure y_start and y_end are within bounds
                y_start = max(0, y_start)
                y_end = min(padded_height - 1, y_end)

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
                    padded_image[y_start:y_end, x:x + rect_width] = rect
                else:
                    cv2.rectangle(padded_image, (x, y_start), (x_end, y_end), color, thickness=-1)
        elif visualization_method == 'line':
            curve_smoothing = kwargs.get('curve_smoothing')

            # Center the visualization
            baseline_y = padded_height // 2
            x_offset = (padded_width - visualization_length) // 2

            # Apply curve smoothing if specified
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

            # X-axis using visualization length
            num_points = len(amplitude)
            x_values = np.linspace(x_offset, x_offset + visualization_length, num_points)

            # Single visualization with direction based on reflect
            if reflect:
                y_values = baseline_y + amplitude
            else:
                y_values = baseline_y - amplitude

            points = np.array([x_values, y_values]).T.astype(np.int32)

            if len(points) > 1:
                cv2.polylines(padded_image, [points], False, (1.0, 1.0, 1.0))

        # Apply rotation if needed
        if rotation != 0:
            # Rotate around center of padded image
            M = cv2.getRotationMatrix2D((padded_width // 2, padded_height // 2), rotation, 1.0)
            padded_image = cv2.warpAffine(padded_image, M, (padded_width, padded_height))

        # Calculate final position
        target_x = int(screen_width * position_x)
        target_y = int(screen_height * position_y)
        
        # Calculate the region to extract from padded image
        start_x = padded_width // 2 - target_x
        start_y = padded_height // 2 - target_y
        
        # Extract the correctly positioned region
        image = padded_image[start_y:start_y + screen_height, start_x:start_x + screen_width]

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
        base_required["feature_param"] = cls.get_modifiable_params()
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

    def apply_effect_internal(self, processor: BaseAudioProcessor, **kwargs):
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

@apply_tooltips
class FlexAudioVisualizerContour(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        base_required["feature_param"] = cls.get_modifiable_params()
        # Remove screen_width, screen_height, position_x, and position_y
        for param in ["screen_width", "screen_height", "position_x", "position_y"]:
            if param in base_required:

                del base_required[param]

        new_inputs = {
            "required": {
                "mask": ("MASK",),  # Input mask to find contour
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                # Parameters common to both methods
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_points": ("INT", {"default": 360, "min": 3, "max": 1000, "step": 1}),
                # Audio processing parameters
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                # Visualization parameters
                "bar_length": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "contour_smoothing": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                # New parameters for direction and multi-contour
                "direction": (["outward", "inward", "both"], {"default": "outward"}),
                "min_contour_area": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 10000.0, "step": 10.0}),
                "max_contours": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "distribute_by": (["area", "perimeter", "equal"], {"default": "perimeter"}),
            }
        }

        required = {**base_required, **new_inputs["required"]}
        optional = base_optional

        return {
            "required": required,
            "optional": optional
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["smoothing", "rotation", "num_points", "fft_size", "min_frequency", "max_frequency", 
                "bar_length", "line_width", "contour_smoothing", "direction", "min_contour_area", 
                "max_contours", "None"]

    def apply_effect(self, audio, frame_rate, mask, strength, feature_param, feature_mode,
                     feature_threshold, opt_feature=None, **kwargs):
        # Get dimensions from mask
        batch_size, screen_height, screen_width = mask.shape
        
        # Add mask to kwargs for the draw method
        kwargs['mask'] = mask
        
        # Call parent with mask dimensions as screen dimensions
        return super().apply_effect(
            audio, frame_rate, screen_width, screen_height,
            strength, feature_param, feature_mode, feature_threshold,
            opt_feature, **kwargs
        )

    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        # Reuse existing audio processing logic
        visualization_feature = kwargs.get('visualization_feature')
        smoothing = kwargs.get('smoothing')
        num_points = kwargs.get('num_points')
        fft_size = kwargs.get('fft_size')
        min_frequency = kwargs.get('min_frequency')
        max_frequency = kwargs.get('max_frequency')

        processor.spectrum, feature_value = self.process_audio_data(
            processor, frame_index, visualization_feature, num_points,
            smoothing, fft_size, min_frequency, max_frequency
        )
        return feature_value

    def apply_effect_internal(self, processor: BaseAudioProcessor, **kwargs):
        # Get parameters
        mask = kwargs.get('mask')
        visualization_method = kwargs.get('visualization_method')
        batch_size, screen_height, screen_width = mask.shape
        line_width = kwargs.get('line_width')
        bar_length = kwargs.get('bar_length')
        contour_smoothing = kwargs.get('contour_smoothing')
        rotation = kwargs.get('rotation', 0.0) % 360.0
        direction = kwargs.get('direction', 'outward')
        min_contour_area = kwargs.get('min_contour_area', 100.0)
        max_contours = kwargs.get('max_contours', 5)
        distribute_by = kwargs.get('distribute_by', 'perimeter')
        
        # Get the frame index from the processor's current state
        frame_index = processor.current_frame if hasattr(processor, 'current_frame') else 0
        
        # Create output image
        image = np.zeros((screen_height, screen_width, 3), dtype=np.float32)
        
        # Use the mask corresponding to the current frame
        frame_index = min(frame_index, batch_size - 1)  # Ensure we don't exceed batch size
        mask_uint8 = (mask[frame_index].numpy() * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image

        # Filter and sort contours by area
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        valid_contours = valid_contours[:max_contours]

        if not valid_contours:
            return image

        # Calculate distribution weights based on chosen method
        if distribute_by == 'area':
            weights = [cv2.contourArea(c) for c in valid_contours]
        elif distribute_by == 'perimeter':
            weights = [cv2.arcLength(c, True) for c in valid_contours]
        else:  # 'equal'
            weights = [1] * len(valid_contours)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]

        # Get spectrum data
        data = processor.spectrum
        
        # Function to process a single contour
        def process_contour(contour, start_idx, end_idx, direction_multiplier=1.0):
            # Apply contour smoothing if specified
            if contour_smoothing > 0:
                epsilon = contour_smoothing * cv2.arcLength(contour, True) * 0.01
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Prepare contour points
            contour = contour.squeeze()
            if len(contour.shape) < 2:  # Handle single-point contours
                return
                
            contour_length = len(contour)
            if contour_length < 2:
                return
                
            # Ensure the contour is closed
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])
                contour_length += 1
            
            # Apply rotation as an offset along the contour
            rotation_offset = int((rotation / 360.0) * contour_length)
            
            # Get the portion of data for this contour
            contour_data = data[start_idx:end_idx]
            num_points = len(contour_data)
            
            # Create points along the contour with rotation offset
            indices = (np.linspace(0, contour_length - 1, num_points) + rotation_offset) % (contour_length - 1)
            
            # Interpolate contour points
            x_coords = np.interp(indices, range(contour_length), contour[:, 0])
            y_coords = np.interp(indices, range(contour_length), contour[:, 1])
            
            # Calculate normals along the contour
            dx = np.gradient(x_coords)
            dy = np.gradient(y_coords)
            
            # Normalize the normals
            lengths = np.sqrt(dx**2 + dy**2)
            lengths = np.where(lengths > 0, lengths, 1.0)
            normals_x = -dy / lengths
            normals_y = dx / lengths
            
            # Replace any NaN values
            normals_x = np.nan_to_num(normals_x, 0.0)
            normals_y = np.nan_to_num(normals_y, 0.0)

            if visualization_method == 'bar':
                # Draw bars along the contour
                for i, amplitude in enumerate(contour_data):
                    x1, y1 = int(x_coords[i]), int(y_coords[i])
                    
                    # Apply direction multiplier to the bar length
                    bar_height = amplitude * bar_length * direction_multiplier
                    x2 = int(x1 + normals_x[i] * bar_height)
                    y2 = int(y1 + normals_y[i] * bar_height)
                    
                    # Clip coordinates to image bounds
                    x1 = np.clip(x1, 0, screen_width - 1)
                    y1 = np.clip(y1, 0, screen_height - 1)
                    x2 = np.clip(x2, 0, screen_width - 1)
                    y2 = np.clip(y2, 0, screen_height - 1)
                    
                    cv2.line(image, (x1, y1), (x2, y2), (1.0, 1.0, 1.0), thickness=line_width)
            
            else:  # line mode
                # Calculate points for the continuous line
                points = np.column_stack([
                    x_coords + normals_x * contour_data * bar_length * direction_multiplier,
                    y_coords + normals_y * contour_data * bar_length * direction_multiplier
                ]).astype(np.int32)
                
                # Clip points to image bounds
                points[:, 0] = np.clip(points[:, 0], 0, screen_width - 1)
                points[:, 1] = np.clip(points[:, 1], 0, screen_height - 1)
                
                # Draw the continuous line
                cv2.polylines(image, [points], True, (1.0, 1.0, 1.0), thickness=line_width)

        # Distribute data points among contours
        total_points = len(data)
        start_idx = 0
        
        for i, (contour, weight) in enumerate(zip(valid_contours, weights)):
            num_points = int(round(total_points * weight))
            if i == len(valid_contours) - 1:  # Last contour gets remaining points
                num_points = total_points - start_idx
            end_idx = start_idx + num_points

            if direction == "both":
                # For "both" direction, process the contour twice with half amplitude
                process_contour(contour, start_idx, end_idx, 0.5)  # Outward
                process_contour(contour, start_idx, end_idx, -0.5)  # Inward
            else:
                # For single direction, process once with full amplitude
                direction_multiplier = -1.0 if direction == "inward" else 1.0
                process_contour(contour, start_idx, end_idx, direction_multiplier)

            start_idx = end_idx
            
        return image

