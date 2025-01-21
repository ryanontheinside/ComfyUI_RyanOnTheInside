from abc import ABC, abstractmethod
import numpy as np
import torch
import cv2
from ..masks.mask_utils import calculate_optical_flow
from scipy.interpolate import interp1d, make_interp_spline
import json
class BaseFeature(ABC):
    def __init__(self, name, feature_type, frame_rate, frame_count, width, height):
        self.name = name
        self.type = feature_type
        self.frame_rate = frame_rate
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.data = None
        self.features = None
        self.inverted = False
        self._min_value = None
        self._max_value = None

    @property
    def min_value(self):
        """Get minimum value - either set explicitly or calculated from data"""
        if self._min_value is not None:
            return self._min_value
        
        if self.data is not None:
            return float(np.min(self.data))
        elif self.features is not None and hasattr(self, 'feature_name'):
            return float(min(self.features[self.feature_name]))
        return 0.0  # Default fallback

    @min_value.setter
    def min_value(self, value):
        """Set minimum value explicitly"""
        self._min_value = float(value)

    @property
    def max_value(self):
        """Get maximum value - either set explicitly or calculated from data"""
        if self._max_value is not None:
            return self._max_value
        
        if self.data is not None:
            return float(np.max(self.data))
        elif self.features is not None and hasattr(self, 'feature_name'):
            return float(max(self.features[self.feature_name]))
        return 1.0  # Default fallback

    @max_value.setter
    def max_value(self, value):
        """Set maximum value explicitly"""
        self._max_value = float(value)

    @abstractmethod
    def extract(self):
        pass

    @classmethod
    @abstractmethod
    def get_extraction_methods(cls):
        """Return a list of parameter names that can be modulated."""
        return []
    
    def get_value_at_frame(self, frame_index):
        if self.data is not None:
            return self.data[frame_index]
        elif self.features is not None and hasattr(self, 'feature_name'):
            return self.features[self.feature_name][frame_index]
        else:
            raise ValueError("No data or features available")
    

    def normalize(self):
        if self.data is not None:
            self.data = (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return self
    

    def get_normalized_data(self):
        if self.data is not None:
            return (self.data - np.min(self.data)) / (np.max(self.data) - np.min(self.data))
        return None
    

    def invert(self):
        if self.data is not None:
            self.data = 1 - self.data
        if self.features is not None:
            for key in self.features:
                if isinstance(self.features[key], list):
                    self.features[key] = [1 - value for value in self.features[key]]
                elif isinstance(self.features[key], np.ndarray):
                    self.features[key] = 1 - self.features[key]
                elif isinstance(self.features[key], torch.Tensor):
                    self.features[key] = 1 - self.features[key]
        self.inverted = not self.inverted
        return self
    
class FloatFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, float_values, feature_type='raw'):
        super().__init__(name, "float", frame_rate, frame_count, width, height)
        self.float_values = np.array(float_values)
        self.feature_type = feature_type
        self.available_features = self.get_extraction_methods()

    @classmethod
    def get_extraction_methods(cls):
        return ["raw", "smooth", "cumulative"]

    def extract(self):
        if len(self.float_values) != self.frame_count:
            # Interpolate to match frame count if necessary
            x = np.linspace(0, 1, len(self.float_values))
            x_new = np.linspace(0, 1, self.frame_count)
            self.float_values = np.interp(x_new, x, self.float_values)

        if self.feature_type == "raw":
            self.data = self.float_values
        elif self.feature_type == "smooth":
            # Apply simple moving average smoothing
            window_size = max(3, self.frame_count // 30)  # Dynamic window size
            kernel = np.ones(window_size) / window_size
            self.data = np.convolve(self.float_values, kernel, mode='same')
        elif self.feature_type == "cumulative":
            # Cumulative sum of values
            self.data = np.cumsum(self.float_values)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

        return self

class ManualFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, start_frame, end_frame, start_value, end_value, method='linear'):
        super().__init__(name, "manual", frame_rate, frame_count, width, height)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_value = start_value
        self.end_value = end_value
        self.method = method

    @classmethod
    def get_extraction_methods(cls):
        return ["manual"]

    def extract(self):
        # Initialize data with zeros
        self.data = np.zeros(self.frame_count)
        
        # Ensure start_frame and end_frame are within bounds
        if 0 <= self.start_frame < self.end_frame <= self.frame_count:
            x = np.array([self.start_frame, self.end_frame])
            y = np.array([self.start_value, self.end_value])

            if self.method == 'linear':
                f = interp1d(x, y, kind='linear', fill_value="extrapolate")
            elif self.method == 'nearest':
                f = interp1d(x, y, kind='nearest', fill_value="extrapolate")
            elif self.method == 'cubic':
                if len(x) >= 4:  # Ensure enough points for cubic
                    f = make_interp_spline(x, y, k=3)
                else:
                    raise ValueError("Cubic interpolation requires at least four data points.")
            elif self.method == 'ease_in':
                f = self.ease_in_interpolation(x, y)
            elif self.method == 'ease_out':
                f = self.ease_out_interpolation(x, y)
            else:
                raise ValueError(f"Unsupported interpolation method: {self.method}")

            # Apply interpolation
            self.data[self.start_frame:self.end_frame] = f(np.arange(self.start_frame, self.end_frame))
        else:
            raise ValueError("Start and end frames must be within the range of frame count.")
        
        return self

    def ease_in_interpolation(self, x, y):
        def ease_in(t):
            return (t - x[0]) / (x[1] - x[0]) ** 2 * (y[1] - y[0]) + y[0]
        return ease_in

    def ease_out_interpolation(self, x, y):
        def ease_out(t):
            return 1 - (1 - (t - x[0]) / (x[1] - x[0])) ** 2 * (y[1] - y[0]) + y[0]
        return ease_out

class TimeFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, effect_type='smooth', speed=1, offset=0.0):
        super().__init__(name, "time", frame_rate, frame_count, width, height)
        self.effect_type = effect_type
        # speed is now in frames (how many frames to complete one cycle)
        self.speed = max(1, int(speed))  # Ensure at least 1 frame
        self.offset = offset

    @classmethod
    def get_extraction_methods(self):
        return [
            "smooth", "accelerate", "pulse", "sawtooth","bounce"
        ]

    def extract(self):
        # Calculate time values based on frames directly
        frames = np.arange(self.frame_count)
        # Convert to cycle position (0 to 1) based on speed in frames
        t = (frames / self.speed + self.offset) % 1

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
        
        return self

class DepthFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, depth_maps, feature_name='mean_depth'):
        super().__init__(name, "depth", frame_rate, frame_count, width, height)
        self.depth_maps = depth_maps
        self.features = None
        self.feature_name = feature_name
        self.available_features = self.get_extraction_methods()

    @classmethod
    def get_extraction_methods(self):
        return [
            "mean_depth", "depth_variance", "depth_range", "gradient_magnitude",
            "foreground_ratio", "midground_ratio", "background_ratio"
        ]

    def extract(self):
        self.features = {self.feature_name: []}

        combined_depth = torch.mean(self.depth_maps, dim=-1)  

        for depth_map in combined_depth:
            depth_min = torch.min(depth_map)
            depth_max = torch.max(depth_map)
            if depth_max > depth_min:
                normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                normalized_depth = torch.zeros_like(depth_map)
            
            if self.feature_name == 'mean_depth':
                self.features[self.feature_name].append(torch.mean(normalized_depth).item())
            elif self.feature_name == 'depth_variance':
                self.features[self.feature_name].append(torch.var(normalized_depth).item())
            elif self.feature_name == 'depth_range':
                self.features[self.feature_name].append((depth_max - depth_min).item())
            elif self.feature_name == 'gradient_magnitude':
                grad_y, grad_x = torch.gradient(normalized_depth)
                gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
                self.features[self.feature_name].append(torch.mean(gradient_magnitude).item())
            elif self.feature_name in ['foreground_ratio', 'midground_ratio', 'background_ratio']:
                total_pixels = normalized_depth.numel()
                foreground = torch.sum(normalized_depth < 0.33).item() / total_pixels
                midground = torch.sum((normalized_depth >= 0.33) & (normalized_depth < 0.66)).item() / total_pixels
                background = torch.sum(normalized_depth >= 0.66).item() / total_pixels
                if self.feature_name == 'foreground_ratio':
                    self.features[self.feature_name].append(foreground)
                elif self.feature_name == 'midground_ratio':
                    self.features[self.feature_name].append(midground)
                elif self.feature_name == 'background_ratio':
                    self.features[self.feature_name].append(background)

        feature_tensor = torch.tensor(self.features[self.feature_name])
        feature_min = torch.min(feature_tensor)
        feature_max = torch.max(feature_tensor)
        if feature_max > feature_min:
            self.features[self.feature_name] = ((feature_tensor - feature_min) / (feature_max - feature_min)).tolist()
        else:
            self.features[self.feature_name] = torch.zeros_like(feature_tensor).tolist()

        return self

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

class ColorFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, images, feature_name='dominant_color'):
        if images.dim() != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected images in BHWC format, but got shape {images.shape}")
        super().__init__(name, "color", frame_rate, frame_count, width, height)
        self.images = images
        self.feature_name = feature_name
        self.available_features = self.get_extraction_methods()

    @classmethod
    def get_extraction_methods(self):
        return [
            "dominant_color", "color_variance", "saturation",
            "red_ratio", "green_ratio", "blue_ratio"
        ]
    
    def extract(self):
        self.features = {self.feature_name: []}
        
        images = self.images.float() / 255.0 if self.images.dtype == torch.uint8 else self.images
        
        if self.feature_name in ['red_ratio', 'green_ratio', 'blue_ratio']:
            color_sums = torch.sum(images, dim=(1, 2))
            totals = torch.sum(color_sums, dim=1, keepdim=True)
            ratios = color_sums / totals
            
            if self.feature_name == 'red_ratio':
                self.features[self.feature_name] = ratios[:, 0].tolist()
            elif self.feature_name == 'green_ratio':
                self.features[self.feature_name] = ratios[:, 1].tolist()
            elif self.feature_name == 'blue_ratio':
                self.features[self.feature_name] = ratios[:, 2].tolist()
        else:
            for image in images:
                self._extract_features(image)

        self._normalize_features()
        print("ColorFeature extraction completed")
        return self

    def _extract_features(self, image):
        if self.feature_name == 'dominant_color':
            self.features[self.feature_name].append(self._get_dominant_color(image))
        elif self.feature_name == 'color_variance':
            self.features[self.feature_name].append(torch.var(image.reshape(-1, 3), dim=0).mean().item())
        elif self.feature_name == 'saturation':
            self.features[self.feature_name].append(self._calculate_saturation(image).item())

    def _get_dominant_color(self, image):
        print("Calculating dominant color")
        quantized = (image * 31).long().reshape(-1, 3)
        unique, counts = np.unique(quantized.cpu().numpy(), axis=0, return_counts=True)
        dominant = unique[np.argmax(counts)]
        return np.mean(dominant / 31)

    def _calculate_saturation(self, image):
        print("Calculating saturation")
        max_val, _ = torch.max(image, dim=-1)
        min_val, _ = torch.min(image, dim=-1)
        return torch.mean(max_val - min_val)

    def _normalize_features(self):
        for key in self.features:
            feature_tensor = torch.tensor(self.features[key])
            feature_min = torch.min(feature_tensor)
            feature_max = torch.max(feature_tensor)
            if feature_max > feature_min:
                self.features[key] = ((feature_tensor - feature_min) / (feature_max - feature_min)).tolist()
            else:
                self.features[key] = torch.zeros_like(feature_tensor).tolist()

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

class BrightnessFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, images, feature_name='mean_brightness'):
        super().__init__(name, "brightness", frame_rate, frame_count, width, height)
        self.images = images
        self.feature_name = feature_name
        self.available_features = self.get_extraction_methods()

    @classmethod
    def get_extraction_methods(self):
        return [
            "mean_brightness", "brightness_variance", "brightness_histogram",
            "dark_ratio", "mid_ratio", "bright_ratio"
        ]
    
    def extract(self):
        self.features = {self.feature_name: []}
        
        images = self.images.float() / 255.0 if self.images.dtype == torch.uint8 else self.images
        
        for image in images:
            self._extract_features(image)

        self._normalize_features()
        return self

    def _extract_features(self, image):
        grayscale = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
        
        if self.feature_name == 'mean_brightness':
            self.features[self.feature_name].append(torch.mean(grayscale).item())
        elif self.feature_name == 'brightness_variance':
            self.features[self.feature_name].append(torch.var(grayscale).item())
        elif self.feature_name == 'brightness_histogram':
            histogram = torch.histc(grayscale, bins=10, min=0, max=1)
            self.features[self.feature_name].append(histogram.tolist())
        else:
            total_pixels = grayscale.numel()
            if self.feature_name == 'dark_ratio':
                self.features[self.feature_name].append((torch.sum(grayscale < 0.3) / total_pixels).item())
            elif self.feature_name == 'mid_ratio':
                self.features[self.feature_name].append((torch.sum((grayscale >= 0.3) & (grayscale < 0.7)) / total_pixels).item())
            elif self.feature_name == 'bright_ratio':
                self.features[self.feature_name].append((torch.sum(grayscale >= 0.7) / total_pixels).item())

    def _normalize_features(self):
        for key in self.features:
            if key != 'brightness_histogram':
                feature_tensor = torch.tensor(self.features[key])
                feature_min = torch.min(feature_tensor)
                feature_max = torch.max(feature_tensor)
                if feature_max > feature_min:
                    self.features[key] = ((feature_tensor - feature_min) / (feature_max - feature_min)).tolist()
                else:
                    self.features[key] = torch.zeros_like(feature_tensor).tolist()

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

class MotionFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, images, feature_name='mean_motion', flow_method='Farneback', flow_threshold=0.0, magnitude_threshold=0.0, progress_callback=None):
        super().__init__(name, "motion", frame_rate, frame_count, width, height)
        self.images = images
        self.feature_name = feature_name
        self.flow_method = flow_method
        self.flow_threshold = flow_threshold
        self.magnitude_threshold = magnitude_threshold
        self.available_features = self.get_extraction_methods()
        self.progress_callback = progress_callback

    @classmethod
    def get_extraction_methods(self):
        return [
            "mean_motion", "max_motion", "motion_direction",
            "horizontal_motion", "vertical_motion", "motion_complexity",
            "motion_speed"
        ]
    
    
    def extract(self):
        print("Starting MotionFeature extraction")
        self.features = {self.feature_name: []}
        
        images_np = (self.images.cpu().numpy() * 255).astype(np.uint8)
        num_frames = len(images_np) - 1

        for i in range(num_frames):
            print(f"Processing frames {i+1} and {i+2}")
            frame1 = images_np[i]
            frame2 = images_np[i+1]
            
            flow = calculate_optical_flow(frame1, frame2, self.flow_method)
            self._extract_features(flow)
            
            if self.progress_callback:
                self.progress_callback(i + 1, num_frames)

        self._pad_last_frame()
        self._normalize_features()
        print("MotionFeature extraction completed")
        return self

    def _extract_features(self, flow):
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        if self.flow_threshold > 0:
            flow_magnitude[flow_magnitude < self.flow_threshold] = 0
        if self.magnitude_threshold > 0:
            flow_magnitude[flow_magnitude < self.magnitude_threshold * np.max(flow_magnitude)] = 0
        
        if self.feature_name == 'mean_motion':
            self.features[self.feature_name].append(np.mean(flow_magnitude))
        elif self.feature_name == 'max_motion':
            self.features[self.feature_name].append(np.max(flow_magnitude))
        elif self.feature_name == 'motion_direction':
            self.features[self.feature_name].append(np.mean(np.arctan2(flow[..., 1], flow[..., 0])))
        elif self.feature_name == 'horizontal_motion':
            self.features[self.feature_name].append(np.mean(np.abs(flow[..., 0])))
        elif self.feature_name == 'vertical_motion':
            self.features[self.feature_name].append(np.mean(np.abs(flow[..., 1])))
        elif self.feature_name == 'motion_complexity':
            self.features[self.feature_name].append(np.std(flow_magnitude))
        elif self.feature_name == 'motion_speed':
            self.features[self.feature_name].append(np.mean(flow_magnitude) * self.frame_rate)

    def _pad_last_frame(self):
        for feature in self.features:
            if feature != "motion_direction":
                self.features[feature].append(self.features[feature][-1])
            else:
                self.features[feature].append(0)

    def _normalize_features(self):
        for key in self.features:
            if key != "motion_direction":
                feature_array = np.array(self.features[key])
                feature_min = np.min(feature_array)
                feature_max = np.max(feature_array)
                if feature_max > feature_min:
                    self.features[key] = ((feature_array - feature_min) / (feature_max - feature_min)).tolist()
                else:
                    self.features[key] = np.zeros_like(feature_array).tolist()
            else:
                self.features[key] = (np.array(self.features[key]) / (2 * np.pi)).tolist()

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

class AreaFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, masks, feature_type='total_area', threshold=0.5):
        super().__init__(name, "area", frame_rate, frame_count, width, height)
        self.masks = masks
        self.feature_type = feature_type
        self.threshold = threshold
        self.available_features = self.get_extraction_methods()

    @classmethod
    def get_extraction_methods(self):
        return ["total_area", "largest_contour", "bounding_box"]

    def extract(self):
        self.data = []
        
        for mask in self.masks:
            mask_np = mask.cpu().numpy()
            binary_mask = (mask_np > self.threshold).astype(np.uint8)
            
            if self.feature_type == 'total_area':
                area = np.sum(binary_mask)
            elif self.feature_type == 'largest_contour':
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                else:
                    area = 0
            elif self.feature_type == 'bounding_box':
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    area = w * h
                else:
                    area = 0
            else:
                raise ValueError(f"Unsupported feature type: {self.feature_type}")
            
            self.data.append(area)
        
        return self

    def normalize(self):
        if self.data:
            min_val = min(self.data)
            max_val = max(self.data)
            if max_val > min_val:
                self.data = [(v - min_val) / (max_val - min_val) for v in self.data]
            else:
                self.data = [0] * len(self.data)
        return self

    def get_value_at_frame(self, frame_index):
        if self.data is not None and 0 <= frame_index < len(self.data):
            return self.data[frame_index]
        else:
            raise ValueError("Invalid frame index or no data available")

    def set_active_feature(self, feature_name):
        if feature_name in self.available_features:
            self.feature_name = feature_name
            self.extract()  # Re-extract the data with the new feature type
        else:
            raise ValueError(f"Invalid feature name. Available features are: {', '.join(self.available_features)}")

class DrawableFeature(BaseFeature):
    """A feature that can be drawn on a graph interface"""
    
    @classmethod
    def get_extraction_methods(cls):
        return ["drawn"]
    
    def __init__(self, name, frame_rate, frame_count, width, height, points, method="linear", min_value=0.0, max_value=1.0, fill_value=0.0):
        super().__init__(name, "drawn", frame_rate, frame_count, width, height)
        self.points = points  # List of (frame, value) tuples
        self.method = method
        self._min_value = float(min_value)
        self._max_value = float(max_value)
        self.fill_value = fill_value
        


    def extract(self):
        """Convert drawn points into a continuous feature curve"""
        if not self.points:
            self.data = np.full(self.frame_count, self.fill_value, dtype=np.float32)
            return self
            
        # Sort points by frame number
        sorted_points = sorted(self.points, key=lambda x: x[0])
        frames, values = zip(*sorted_points)
        frames = np.array(frames)
        values = np.array(values)
        x = np.arange(self.frame_count)
        
        # Initialize with fill value
        self.data = np.full(self.frame_count, self.fill_value, dtype=np.float32)
        
        if len(frames) == 1:
            # Single point - just set that point
            self.data[int(frames[0])] = values[0]
        else:
            # Multiple points - interpolate based on method
            if self.method == "linear":
                f = interp1d(frames, values, kind='linear', bounds_error=False, fill_value=self.fill_value)
                self.data = f(x).astype(np.float32)
            
            elif self.method == "cubic":
                if len(frames) >= 4:
                    from scipy.interpolate import CubicSpline
                    f = CubicSpline(frames, values, bc_type='natural')
                    mask = (x >= frames[0]) & (x <= frames[-1])
                    self.data[mask] = f(x[mask]).astype(np.float32)
                else:
                    # Fall back to linear if not enough points
                    f = interp1d(frames, values, kind='linear', bounds_error=False, fill_value=self.fill_value)
                    self.data = f(x).astype(np.float32)
            
            elif self.method == "nearest":
                f = interp1d(frames, values, kind='nearest', bounds_error=False, fill_value=self.fill_value)
                self.data = f(x).astype(np.float32)
            
            elif self.method == "zero":
                # Only set values at exact points
                for frame, value in zip(frames, values):
                    self.data[int(frame)] = value
            
            elif self.method == "hold":
                # Hold each value until the next point
                mask = (x >= frames[0]) & (x <= frames[-1])
                for i in range(len(frames)-1):
                    self.data[int(frames[i]):int(frames[i+1])] = values[i]
                self.data[int(frames[-1])] = values[-1]
            
            elif self.method == "ease_in":
                # Quadratic ease-in
                mask = (x >= frames[0]) & (x <= frames[-1])
                t = np.zeros_like(x, dtype=float)
                t[mask] = (x[mask] - frames[0]) / (frames[-1] - frames[0])
                f = interp1d(frames, values, kind='linear', bounds_error=False, fill_value=self.fill_value)
                self.data[mask] = (t[mask] * t[mask] * f(x[mask])).astype(np.float32)
            
            elif self.method == "ease_out":
                # Quadratic ease-out
                mask = (x >= frames[0]) & (x <= frames[-1])
                t = np.zeros_like(x, dtype=float)
                t[mask] = (x[mask] - frames[0]) / (frames[-1] - frames[0])
                f = interp1d(frames, values, kind='linear', bounds_error=False, fill_value=self.fill_value)
                self.data[mask] = ((2 - t[mask]) * t[mask] * f(x[mask])).astype(np.float32)
            
            else:
                raise ValueError(f"Unsupported interpolation method: {self.method}")
        
        # Normalize the data to 0-1 range
        if self.max_value > self.min_value:
            self.data = (self.data - self.min_value) / (self.max_value - self.min_value)
        
        return self

class WhisperFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, width, height, alignment_data, trigger_pairs=None, feature_name='word_timing'):
        super().__init__(name, "whisper", frame_rate, frame_count, width, height)
        # Handle both string and dict/list alignment data
        if isinstance(alignment_data, str):
            try:
                self.alignment_data = json.loads(alignment_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid alignment data format: must be valid JSON string or dict/list")
        else:
            self.alignment_data = alignment_data
        self.trigger_pairs = trigger_pairs  # Now a TriggerSet instance
        self.feature_name = feature_name
        self.available_features = self.get_extraction_methods()

    @classmethod
    def get_extraction_methods(cls):
        return [
            "trigger_values",   # Uses trigger pairs to create value sequences
            "word_timing",      # Creates peaks at word starts
            "segment_timing",   # Creates plateaus during segments
            "speech_density",   # Measures words per second over time
            "silence_ratio"     # Ratio of silence vs speech
        ]

    def extract(self):
        self.features = {self.feature_name: [0.0] * self.frame_count}
        
        if self.feature_name == "word_timing":
            self._extract_word_timing()
        elif self.feature_name == "segment_timing":
            self._extract_segment_timing()
        elif self.feature_name == "trigger_values":
            self._extract_trigger_values()
        elif self.feature_name == "speech_density":
            self._extract_speech_density()
        elif self.feature_name == "silence_ratio":
            self._extract_silence_ratio()
        
        self._normalize_features()
        return self

    def _extract_word_timing(self):
        feature_data = [0.0] * self.frame_count
        for item in self.alignment_data:
            start_frame = int(item["start"] * self.frame_rate)
            if 0 <= start_frame < self.frame_count:
                feature_data[start_frame] = 1.0
        self.features[self.feature_name] = feature_data

    def _extract_segment_timing(self):
        feature_data = [0.0] * self.frame_count
        for item in self.alignment_data:
            start_frame = int(item["start"] * self.frame_rate)
            end_frame = int(item["end"] * self.frame_rate)
            for frame in range(max(0, start_frame), min(end_frame + 1, self.frame_count)):
                feature_data[frame] = 1.0
        self.features[self.feature_name] = feature_data

    def _extract_trigger_values(self):
        if not self.trigger_pairs:
            raise ValueError("Trigger pairs required for trigger_values feature")
            
        feature_data = [0.0] * self.frame_count
        accumulated_data = [0.0] * self.frame_count
        
        for item in self.alignment_data:
            text = item["value"].lower()
            start_frame = int(item["start"] * self.frame_rate)
            end_frame = int(item["end"] * self.frame_rate)
            
            for trigger in self.trigger_pairs.triggers:  # Access triggers directly from TriggerSet
                pattern = trigger["pattern"].lower()
                if pattern in text:
                    start_val, end_val = trigger["values"]
                    blend_mode = trigger.get("blend_mode", "blend")
                    
                    # Create temporary array for this trigger's values
                    trigger_data = [0.0] * self.frame_count
                    if start_frame < self.frame_count:
                        trigger_data[start_frame] = start_val
                    if end_frame < self.frame_count:
                        trigger_data[end_frame] = end_val
                    
                    # Blend with accumulated result based on blend mode
                    for i in range(self.frame_count):
                        if blend_mode == "blend":
                            accumulated_data[i] = (accumulated_data[i] + trigger_data[i]) / 2
                        elif blend_mode == "add":
                            accumulated_data[i] = min(1.0, accumulated_data[i] + trigger_data[i])
                        elif blend_mode == "multiply":
                            accumulated_data[i] = accumulated_data[i] * trigger_data[i]
                        elif blend_mode == "max":
                            accumulated_data[i] = max(accumulated_data[i], trigger_data[i])
        
        self.features[self.feature_name] = accumulated_data

    def _extract_speech_density(self):
        feature_data = [0.0] * self.frame_count
        window_size = int(self.frame_rate)  # 1-second window
        
        for item in self.alignment_data:
            start_frame = int(item["start"] * self.frame_rate)
            if 0 <= start_frame < self.frame_count:
                # Add word density over a window
                for i in range(max(0, start_frame - window_size), min(start_frame + window_size, self.frame_count)):
                    feature_data[i] += 1.0 / (2 * window_size)
        
        self.features[self.feature_name] = feature_data

    def _extract_silence_ratio(self):
        feature_data = [1.0] * self.frame_count  # Initialize as silence
        
        for item in self.alignment_data:
            start_frame = int(item["start"] * self.frame_rate)
            end_frame = int(item["end"] * self.frame_rate)
            for frame in range(max(0, start_frame), min(end_frame + 1, self.frame_count)):
                feature_data[frame] = 0.0  # Mark as speech
        
        self.features[self.feature_name] = feature_data

    def _normalize_features(self):
        if self.feature_name != "trigger_values":  # Don't normalize trigger values
            feature_array = np.array(self.features[self.feature_name])
            feature_min = np.min(feature_array)
            feature_max = np.max(feature_array)
            if feature_max > feature_min:
                self.features[self.feature_name] = ((feature_array - feature_min) / 
                                                  (feature_max - feature_min)).tolist()

    def get_feature_sequence(self, feature_name=None):
        if self.features is None:
            self.extract()
        if feature_name is None:
            feature_name = self.feature_name
        return self.features.get(feature_name, None)

    def set_active_feature(self, feature_name):
        if feature_name in self.available_features:
            self.feature_name = feature_name
            self.extract()
        else:
            raise ValueError(f"Invalid feature name. Available features are: {', '.join(self.available_features)}")

    def find_trigger_start_time(self, pattern):
        """Find the start time of when a pattern appears in the alignment data"""
        for segment in self.alignment_data:
            if pattern.lower() in segment["value"].lower():
                return segment["start"]
        return None

    def find_trigger_end_time(self, pattern):
        """Find the end time of when a pattern appears in the alignment data"""
        for segment in self.alignment_data:
            if pattern.lower() in segment["value"].lower():
                return segment["end"]
        return None

    def sort_triggers_by_occurrence(self, triggers):
        """Sort triggers based on when their patterns appear in the text"""
        def find_first_occurrence(pattern):
            text = " ".join(segment["value"] for segment in self.alignment_data)
            return text.lower().find(pattern.lower())
        
        return sorted(triggers, key=lambda t: find_first_occurrence(t["pattern"]))

    def get_trigger_frames(self, pattern):
        """Get the frame range where a trigger pattern occurs"""
        start_time = self.find_trigger_start_time(pattern)
        if start_time is None:
            return None, None
            
        end_time = self.find_trigger_end_time(pattern)
        start_frame = int(start_time * self.frame_rate)
        end_frame = int(end_time * self.frame_rate)
        
        return start_frame, end_frame

    def find_all_trigger_frames(self, pattern):
        """Find all frame ranges where a trigger pattern occurs"""
        frame_ranges = []
        pattern = pattern.lower()
        
        for segment in self.alignment_data:
            if pattern in segment["value"].lower():
                start_frame = int(segment["start"] * self.frame_rate)
                end_frame = int(segment["end"] * self.frame_rate)
                frame_ranges.append((start_frame, end_frame))
                
        return frame_ranges

#TODO volume feature

