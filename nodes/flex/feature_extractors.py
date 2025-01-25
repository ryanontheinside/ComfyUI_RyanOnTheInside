from .features import ManualFeature, TimeFeature, DepthFeature, ColorFeature, BrightnessFeature, MotionFeature, AreaFeature, BaseFeature, DrawableFeature, FloatFeature
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d
import json
from ...tooltips import apply_tooltips
from ... import ProgressMixin



@apply_tooltips
class FeatureExtractorBase(ProgressMixin, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {            
            "required": {
                "extraction_method": (["error_not_implemented"],),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "frame_count": ("INT", {"default": 30, "min": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
            }
        }
    
    CATEGORY="RyanOnTheInside/FlexFeatures/Sources"

@apply_tooltips
class FloatFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (FloatFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "floats": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,  "forceInput": True}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, floats, frame_rate, frame_count, width, height, extraction_method):
        values = floats if isinstance(floats, list) else [floats]

        float_feature = FloatFeature("float_feature", frame_rate, frame_count, width, height, values, extraction_method)
        float_feature.extract()
        return (float_feature,)
    
@apply_tooltips
class ManualFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (ManualFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "frame_numbers": ("STRING", {"default": "0,10,20"}),
                "values": ("STRING", {"default": "0.0,0.5,1.0"}),
                "last_value": ("FLOAT", {"default": 1.0}),
                "interpolation_method": (["none", "linear", "ease_in", "ease_out"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"
    CATEGORY = f"{FeatureExtractorBase.CATEGORY}/Manual"

    def _apply_interpolation(self, frame_count, frame_rate, frame_numbers, values, last_value, interpolation_method, width=None, height=None):
        """Shared interpolation logic"""
        manual_feature = ManualFeature("manual_feature", frame_rate, frame_count, width, height,
                                   frame_numbers[0], frame_numbers[-1], values[0], values[-1], 
                                   method=interpolation_method)

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
        manual_feature = self._apply_interpolation(frame_count, frame_rate, frame_numbers, values, last_value, interpolation_method, width, height)
        
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
    CATEGORY = f"{FeatureExtractorBase.CATEGORY}/Manual"

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
class DrawableFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (DrawableFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "points": ("STRING", {"default": "[]"}),  # JSON string of points
                "min_value": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "interpolation_method": (["linear", "cubic", "nearest", "zero", "hold", "ease_in", "ease_out"], {"default": "linear"}),
                "fill_value": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"
    CATEGORY = f"{FeatureExtractorBase.CATEGORY}/Manual"
    
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
            width,
            height,
            point_data,
            method=interpolation_method,
            min_value=min_value,
            max_value=max_value,
            fill_value=fill_value
        )
        drawable_feature.extract()
        return (drawable_feature,)
   
@apply_tooltips
class TimeFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (TimeFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "frames_per_cycle": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1}),
                "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, extraction_method, frames_per_cycle, offset, frame_rate, frame_count, width, height):
        time_feature = TimeFeature("time_effect", frame_rate, frame_count, width, height,
                                effect_type=extraction_method, speed=frames_per_cycle, offset=offset)
        time_feature.extract()
        return (time_feature,)
    
@apply_tooltips
class DepthFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs.pop("frame_count", None)
        parent_inputs["extraction_method"] = (DepthFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "depth_maps": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, depth_maps, frame_rate, width, height, extraction_method):
        frame_count = len(depth_maps)
        depth_feature = DepthFeature("depth_feature", frame_rate, frame_count, width, height, depth_maps, extraction_method)
        depth_feature.extract()
        return (depth_feature,)

@apply_tooltips
class ColorFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs.pop("frame_count", None)
        parent_inputs["extraction_method"] = (ColorFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, images, frame_rate, width, height, extraction_method):
        frame_count = len(images)
        color_feature = ColorFeature("color_feature", frame_rate, frame_count, width, height, images, extraction_method)
        color_feature.extract()
        return (color_feature,)

@apply_tooltips
class BrightnessFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs.pop("frame_count", None)
        parent_inputs["extraction_method"] = (BrightnessFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, images, frame_rate, width, height, extraction_method):
        frame_count = len(images)
        brightness_feature = BrightnessFeature("brightness_feature", frame_rate, frame_count, width, height, images, extraction_method)
        brightness_feature.extract()
        return (brightness_feature,)
    
@apply_tooltips
class MotionFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs.pop("frame_count", None)
        parent_inputs["extraction_method"] = (MotionFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "images": ("IMAGE",),
                "flow_method": (["Farneback", "LucasKanade", "PyramidalLK"],),
                "flow_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "magnitude_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, images, frame_rate, width, height, extraction_method, flow_method, flow_threshold, magnitude_threshold):
        # Use length of images as frame_count
        frame_count = len(images)
        
        def progress_callback(current_step, total_steps):
            self.update_progress(current_step - self.current_progress)

        self.start_progress(frame_count, desc="Extracting motion features")

        motion_feature = MotionFeature(
            "motion_feature",
            frame_rate,
            frame_count,
            width,
            height,
            images,
            extraction_method,
            flow_method,
            flow_threshold,
            magnitude_threshold,
            progress_callback=progress_callback
        )

        motion_feature.extract()
        self.end_progress()

        return (motion_feature,)
    
@apply_tooltips
class AreaFeatureNode(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs.pop("frame_count", None)
        parent_inputs["extraction_method"] = (AreaFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "masks": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "create_feature"

    def create_feature(self, masks, frame_rate, width, height, extraction_method, threshold):
        frame_count = len(masks)
        area_feature = AreaFeature("area_feature", frame_rate, frame_count, width, height, masks, extraction_method, threshold)
        area_feature.extract()
        return (area_feature,)
        
    
@apply_tooltips
class FeatureInfoNode():
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

    RETURN_TYPES = ("STRING", "STRING", "INT","FLOAT", "INT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("name", "type", "frame_rate","frame_rate_float", "frame_count", "width", "height", "min_value", "max_value")
    FUNCTION = "get_info"
    CATEGORY = "RyanOnTheInside/FlexFeatures/Utilities"

    def get_info(self, feature):
        """Extract common information from the feature"""


        #TODO: rename attr to thresholds impending
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]        # Use feature.min_value and feature.max_value if available, otherwise use actual min/max
        min_val = getattr(feature, 'min_value', min(values))
        max_val = getattr(feature, 'max_value', max(values))

        return (
            feature.name,
            feature.type,
            feature.frame_rate,
            float(feature.frame_rate),
            feature.frame_count,
            feature.width if feature.width is not None else 0,
            feature.height if feature.height is not None else 0,
            min_val,
            max_val
        )




