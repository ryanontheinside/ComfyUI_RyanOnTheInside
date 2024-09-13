from abc import ABC, abstractmethod
import numpy as np
import torch
import cv2
from ..masks.mask_utils import calculate_optical_flow

class BaseFeature(ABC):
    def __init__(self, name, feature_type, frame_rate, frame_count):
        self.name = name
        self.type = feature_type
        self.frame_rate = frame_rate
        self.frame_count = frame_count
        self.data = None
        self.features = None
        self.inverted = False

    @abstractmethod
    def extract(self):
        pass

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

class TimeFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, effect_type='smooth', speed=1.0, offset=0.0):
        self.effect_type = effect_type
        self.speed = speed
        self.offset = offset
        super().__init__(name, "time", frame_rate, frame_count)

    def extract(self):
        t = np.linspace(0, self.frame_count / self.frame_rate, self.frame_count)
        t = (t * self.speed + self.offset) % 1

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
        
        return self.normalize()

class DepthFeature(BaseFeature):
    def __init__(self, name, frame_rate, frame_count, depth_maps, feature_name='mean_depth'):
        self.depth_maps = depth_maps
        self.features = None
        self.feature_name = feature_name
        self.available_features = [
            "mean_depth", "depth_variance", "depth_range", "gradient_magnitude",
            "foreground_ratio", "midground_ratio", "background_ratio"
        ]
        super().__init__(name, "depth", frame_rate, frame_count)

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
    def __init__(self, name, frame_rate, frame_count, images, feature_name='dominant_color'):
        if images.dim() != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected images in BHWC format, but got shape {images.shape}")
        self.images = images
        self.feature_name = feature_name
        self.available_features = [
            "dominant_color", "color_variance", "saturation",
            "red_ratio", "green_ratio", "blue_ratio"
        ]
        super().__init__(name, "color", frame_rate, frame_count)

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
    def __init__(self, name, frame_rate, frame_count, images, feature_name='mean_brightness'):
        self.images = images
        self.feature_name = feature_name
        self.available_features = [
            "mean_brightness", "brightness_variance", "brightness_histogram",
            "dark_ratio", "mid_ratio", "bright_ratio"
        ]
        super().__init__(name, "brightness", frame_rate, frame_count)

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
    def __init__(self, name, frame_rate, frame_count, images, feature_name='mean_motion', flow_method='Farneback', flow_threshold=0.0, magnitude_threshold=0.0, progress_callback=None):
        self.images = images
        self.feature_name = feature_name
        self.flow_method = flow_method
        self.flow_threshold = flow_threshold
        self.magnitude_threshold = magnitude_threshold
        self.available_features = [
            "mean_motion", "max_motion", "motion_direction",
            "horizontal_motion", "vertical_motion", "motion_complexity",
            "motion_speed"
        ]
        self.progress_callback = progress_callback
        super().__init__(name, "motion", frame_rate, frame_count)

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
    def __init__(self, name, frame_rate, frame_count, masks, feature_type='total_area', threshold=0.5):
        self.masks = masks
        self.feature_type = feature_type
        self.threshold = threshold
        self.available_features = ["total_area", "largest_contour", "bounding_box"]
        super().__init__(name, "area", frame_rate, frame_count)

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
        
        return self.normalize()

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
            self.feature_type = feature_name
            self.extract()  # Re-extract the data with the new feature type
        else:
            raise ValueError(f"Invalid feature name. Available features are: {', '.join(self.available_features)}")
#TODO volume feature
