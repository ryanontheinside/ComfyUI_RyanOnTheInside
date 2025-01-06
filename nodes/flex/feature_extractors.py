from .feature_pipe import FeaturePipe
from ... import RyanOnTheInside
from .features import ManualFeature, TimeFeature, DepthFeature, ColorFeature, BrightnessFeature, MotionFeature, AreaFeature, BaseFeature, DrawableFeature
from abc import ABC, abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
import typing
import numpy as np
import torch
from scipy.interpolate import interp1d
import json
from ...tooltips import apply_tooltips

@apply_tooltips
class FeatureExtractorBase(RyanOnTheInside, ABC):
    @classmethod
    @abstractmethod
    def feature_type(cls) -> type[BaseFeature]:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        feature_class = cls.feature_type()
        extraction_methods = feature_class.get_extraction_methods()
        return {            
            "required": {
                "extraction_method": (extraction_methods, {"default": extraction_methods[0]}),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "frame_count": ("INT", {"default": 30, "min": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            }
        }

    def __init__(self):
        self.progress_bar = None
        self.tqdm_bar = None
        self.current_progress = 0
        self.total_steps = 0

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)
        self.tqdm_bar = tqdm(total=total_steps, desc=desc, leave=False)
        self.current_progress = 0
        self.total_steps = total_steps

    def update_progress(self, step=1):
        self.current_progress += step
        if self.progress_bar:
            self.progress_bar.update(step)
        if self.tqdm_bar:
            self.tqdm_bar.update(step)

    def end_progress(self):
        if self.tqdm_bar:
            self.tqdm_bar.close()
        self.progress_bar = None
        self.tqdm_bar = None
        self.current_progress = 0
        self.total_steps = 0
    
    CATEGORY="RyanOnTheInside/FlexFeatures"

@apply_tooltips
class ManualFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return ManualFeature  

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "frame_numbers": ("STRING", {"default": "0,10,20"}),
                "values": ("STRING", {"default": "0.0,0.5,1.0"}),
                "last_value": ("FLOAT", {"default": 1.0}),
                "interpolation_method": (["none", "linear", "ease_in", "ease_out"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def _apply_interpolation(self, frame_count, frame_rate, frame_numbers, values, last_value, interpolation_method, height=None, width=None):
        """Shared interpolation logic"""
        manual_feature = ManualFeature("manual_feature", frame_rate, frame_count,
                                   frame_numbers[0], frame_numbers[-1], values[0], values[-1], 
                                   method=interpolation_method, height=height, width=width)

        manual_feature.data = np.zeros(frame_count, dtype=np.float32)
        interpolation_kind = 'linear' if len(frame_numbers) < 3 else 'quadratic'

        if interpolation_method == 'none':
            for frame, value in zip(frame_numbers, values):
                if 0 <= frame < frame_count:
                    manual_feature.data[frame] = value
                else:
                    raise ValueError(f"Frame number {frame} is out of bounds.")
        else:
            if interpolation_method == 'linear':
                f = interp1d(frame_numbers, values, kind='linear', fill_value="extrapolate")
            elif interpolation_method == 'ease_in':
                f = interp1d(frame_numbers, values, kind=interpolation_kind, fill_value="extrapolate")
            elif interpolation_method == 'ease_out':
                reversed_values = [values[-1] - (v - values[0]) for v in values]
                f = interp1d(frame_numbers, reversed_values, kind=interpolation_kind, fill_value="extrapolate")
            
            x = np.arange(frame_count)
            manual_feature.data = f(x)
            
            if interpolation_method == 'ease_out':
                manual_feature.data = values[-1] - (manual_feature.data - values[0])

        return manual_feature

    def create_feature(self, frame_rate, frame_count, frame_numbers, values, last_value, width, height, interpolation_method, extraction_method):
        # Parse inputs
        frame_numbers = list(map(int, frame_numbers.split(',')))
        values = list(map(float, values.split(',')))

        if len(frame_numbers) != len(values):
            raise ValueError("The number of frame numbers must match the number of values.")

        # Validate frame numbers against frame_count
        if max(frame_numbers) >= frame_count:
            raise ValueError(f"Frame numbers must be less than frame_count ({frame_count})")

        # Apply interpolation
        manual_feature = self._apply_interpolation(frame_count, frame_rate, frame_numbers, values, last_value, interpolation_method, height, width)
        
        return (manual_feature,)

@apply_tooltips
class ManualFeatureFromPipe(ManualFeatureNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature_pipe": ("FEATURE_PIPE",),
                "frame_numbers": ("STRING", {"default": "0,10,20"}),
                "values": ("STRING", {"default": "0.0,0.5,1.0"}),
                "last_value": ("FLOAT", {"default": 1.0}),
                "interpolation_method": (["none", "linear", "ease_in", "ease_out"], {"default": "none"}),
            }
        }

    FUNCTION = "create_feature_from_pipe"

    def create_feature_from_pipe(self, feature_pipe, frame_numbers, values, last_value, interpolation_method):
        # Parse inputs
        frame_numbers = list(map(int, frame_numbers.split(',')))
        values = list(map(float, values.split(',')))

        if len(frame_numbers) != len(values):
            raise ValueError("The number of frame numbers must match the number of values.")

        # Append the last frame index and value
        frame_numbers.append(feature_pipe.frame_count - 1)
        values.append(last_value)

        # Apply interpolation using parent class method
        manual_feature = self._apply_interpolation(feature_pipe.frame_count, feature_pipe.frame_rate, frame_numbers, values, last_value, interpolation_method)

        return (manual_feature, feature_pipe)

   
@apply_tooltips
class TimeFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return TimeFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, extraction_method, speed, offset, frame_rate, frame_count, width, height):
        time_feature = TimeFeature("time_effect", frame_rate, frame_count, width, height,
                                effect_type=extraction_method, speed=speed, offset=offset)
        time_feature.extract()
        return (time_feature,)
    
@apply_tooltips
class DepthFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return DepthFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "depth_maps": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, depth_maps, frame_rate, frame_count, width, height, extraction_method):
        depth_feature = DepthFeature("depth_feature", frame_rate, frame_count, depth_maps, extraction_method, width=width, height=height)
        depth_feature.extract()
        return (depth_feature,)

@apply_tooltips
class ColorFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return ColorFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, images, frame_rate, frame_count, width, height, extraction_method):
        color_feature = ColorFeature("color_feature", frame_rate, frame_count, images, extraction_method, width=width, height=height)
        color_feature.extract()
        return (color_feature,)

@apply_tooltips
class BrightnessFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return BrightnessFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, images, frame_rate, frame_count, width, height, extraction_method):
        brightness_feature = BrightnessFeature("brightness_feature", frame_rate, frame_count, images, extraction_method, width=width, height=height)
        brightness_feature.extract()
        return (brightness_feature,)
    
@apply_tooltips
class MotionFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return MotionFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "images": ("IMAGE",),
                "flow_method": (["Farneback", "LucasKanade", "PyramidalLK"],),
                "flow_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "magnitude_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, images, frame_rate, frame_count, width, height, extraction_method, flow_method, flow_threshold, magnitude_threshold):
        def progress_callback(current_step, total_steps):
            self.update_progress(current_step - self.current_progress)

        self.start_progress(frame_count, desc="Extracting motion features")

        motion_feature = MotionFeature(
            "motion_feature",
            frame_rate,
            frame_count,
            images,
            extraction_method,
            flow_method,
            flow_threshold,
            magnitude_threshold,
            width=width,
            height=height,
            progress_callback=progress_callback
        )

        motion_feature.extract()
        self.end_progress()

        return (motion_feature,)
    
@apply_tooltips
class AreaFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return AreaFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "masks": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, masks, frame_rate, frame_count, width, height, extraction_method, threshold):
        area_feature = AreaFeature("area_feature", frame_rate, frame_count, masks, extraction_method, threshold, width=width, height=height)
        area_feature.extract()
        return (area_feature,)
    
@apply_tooltips
class DrawableFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return DrawableFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "points": ("STRING", {"default": "[]"}),  # JSON string of points
                "min_value": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "interpolation_method": (["linear", "cubic", "nearest", "zero", "hold", "ease_in", "ease_out"], {"default": "linear"}),
                "fill_value": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, points, frame_rate, frame_count, width, height, extraction_method, interpolation_method, min_value, max_value, fill_value):
        try:
            point_data = json.loads(points)
            if not isinstance(point_data, list):
                raise ValueError("Points data must be a list")
            # Validate points format
            for point in point_data:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("Each point must be a [frame, value] pair")
                if not (isinstance(point[0], (int, float)) and isinstance(point[1], (float))):
                    raise ValueError("Frame must be number, value must be float")
                if point[0] < 0 or point[0] > frame_count:
                    raise ValueError(f"Frame {point[0]} out of bounds")
                if point[1] < min_value or point[1] > max_value:
                    raise ValueError(f"Value {point[1]} outside range [{min_value}, {max_value}]")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for points")

        drawable_feature = DrawableFeature(
            "drawable_feature",
            frame_rate,
            frame_count,
            point_data,
            method=interpolation_method,
            min_value=min_value,
            max_value=max_value,
            width=width,
            height=height,
            fill_value=fill_value
        )
        drawable_feature.extract()
        return (drawable_feature,)
    
    
@apply_tooltips
class FeatureInfoNode(RyanOnTheInside):
    """
    Node that extracts common information from feature inputs.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),  # Accepts any feature type
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("name", "type", "frame_rate", "frame_count", "width", "height")
    FUNCTION = "get_info"
    CATEGORY = "Ryan/Features"

    def get_info(self, feature):
        """Extract common information from the feature"""
        return (
            feature.name,
            feature.type,
            feature.frame_rate,
            feature.frame_count,
            feature.width if feature.width is not None else 0,
            feature.height if feature.height is not None else 0
        )




