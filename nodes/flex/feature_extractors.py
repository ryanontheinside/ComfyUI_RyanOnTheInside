from .feature_pipe import FeaturePipe
from ... import RyanOnTheInside
from .features import ManualFeature, TimeFeature, DepthFeature, ColorFeature, BrightnessFeature, MotionFeature, AreaFeature, BaseFeature
from abc import ABC, abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
import typing
import numpy as np
import torch
from scipy.interpolate import interp1d

class FeatureExtractorBase(RyanOnTheInside, ABC):
    @classmethod
    @abstractmethod
    def feature_type(cls) -> type[BaseFeature]:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        feature_class = cls.feature_type()
        return {            
            "required": {
                "extraction_method": (feature_class.get_extraction_methods(), {"default": feature_class.get_extraction_methods()[0]}),
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

class ManualFeatureNode(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return ManualFeature  

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "frame_count": ("INT", {"default": 30, "min": 1}),
                "frame_numbers": ("STRING", {"default": "0,10,20"}),
                "values": ("STRING", {"default": "0.0,0.5,1.0"}),
                "last_value": ("FLOAT", {"default": 1.0}),
                "width": ("INT", {"default": 1920, "min": 1}),
                "height": ("INT", {"default": 1080, "min": 1}),
                "interpolation_method": (["none", "linear", "ease_in", "ease_out"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "create_feature"

    def _apply_interpolation(self, feature_pipe, frame_numbers, values, last_value, interpolation_method):
        """Shared interpolation logic"""
        manual_feature = ManualFeature("manual_feature", feature_pipe.frame_rate, feature_pipe.frame_count,
                                     frame_numbers[0], frame_numbers[-1], values[0], values[-1], 
                                     method=interpolation_method)

        manual_feature.data = np.zeros(feature_pipe.frame_count, dtype=np.float32)
        interpolation_kind = 'linear' if len(frame_numbers) < 3 else 'quadratic'

        if interpolation_method == 'none':
            for frame, value in zip(frame_numbers, values):
                if 0 <= frame < feature_pipe.frame_count:
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
            
            x = np.arange(feature_pipe.frame_count)
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

        # Create feature pipe with specified frame_count
        video_frames = torch.zeros((frame_count, height, width, 3), dtype=torch.float32)
        feature_pipe = FeaturePipe(frame_rate, video_frames)

        # Apply interpolation
        manual_feature = self._apply_interpolation(feature_pipe, frame_numbers, values, last_value, interpolation_method)
        
        return (manual_feature, feature_pipe)

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
        manual_feature = self._apply_interpolation(feature_pipe, frame_numbers, values, last_value, interpolation_method)

        return (manual_feature, feature_pipe)

class FirstFeature(FeatureExtractorBase):

    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {            
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "video_frames": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "create_feature"
   
class TimeFeatureNode(FirstFeature):
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

   

    def create_feature(self, extraction_method, speed, offset, video_frames, frame_rate):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        time_feature = TimeFeature("time_effect", feature_pipe.frame_rate, feature_pipe.frame_count, 
                                   effect_type=extraction_method, speed=speed, offset=offset)
        time_feature.extract()
        return (time_feature, feature_pipe)
    
class DepthFeatureNode(FirstFeature):
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

    def create_feature(self, depth_maps, frame_rate, video_frames, extraction_method):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        depth_feature = DepthFeature("depth_feature", feature_pipe.frame_rate, feature_pipe.frame_count, depth_maps, extraction_method)
        depth_feature.extract()
        return (depth_feature, feature_pipe)

class ColorFeatureNode(FirstFeature):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return ColorFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
            }
        }

    def create_feature(self, video_frames, frame_rate, extraction_method):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        color_feature = ColorFeature("color_feature", feature_pipe.frame_rate, feature_pipe.frame_count, video_frames, extraction_method)
        color_feature.extract()
        return (color_feature, feature_pipe)

class BrightnessFeatureNode(FirstFeature):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return BrightnessFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
            }
        }

    def create_feature(self, video_frames, frame_rate, extraction_method):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        brightness_feature = BrightnessFeature("brightness_feature", feature_pipe.frame_rate, feature_pipe.frame_count, video_frames, extraction_method)
        brightness_feature.extract()
        return (brightness_feature, feature_pipe)
    
class MotionFeatureNode(FirstFeature):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return MotionFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "flow_method": (["Farneback", "LucasKanade", "PyramidalLK"],),
                "flow_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "magnitude_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "create_feature"

    def create_feature(self, video_frames, frame_rate, extraction_method, flow_method, flow_threshold, magnitude_threshold):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        num_frames = feature_pipe.frame_count

        self.start_progress(num_frames, desc="Extracting motion features")

        def progress_callback(current_step, total_steps):
            self.update_progress(current_step - self.current_progress)

        motion_feature = MotionFeature(
            "motion_feature",
            feature_pipe.frame_rate,
            feature_pipe.frame_count,
            video_frames,
            extraction_method,
            flow_method,
            flow_threshold,
            magnitude_threshold,
            progress_callback=progress_callback
        )

        motion_feature.extract()
        self.end_progress()

        return (motion_feature, feature_pipe)
    
class AreaFeatureNode(FirstFeature):
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

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "create_feature"

    def create_feature(self, masks, video_frames, frame_rate, extraction_method, threshold):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        area_feature = AreaFeature("area_feature", feature_pipe.frame_rate, feature_pipe.frame_count, 
                                   masks, feature_type=extraction_method, threshold=threshold)
        area_feature.extract()
        return (area_feature, feature_pipe)
    





