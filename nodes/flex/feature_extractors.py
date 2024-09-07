from .feature_pipe import FeaturePipe
from ... import RyanOnTheInside
from .features import AudioFeature, TimeFeature, DepthFeature, ColorFeature, BrightnessFeature, MotionFeature
from .proximity_feature  import  Location
from tqdm import tqdm
from comfy.utils import ProgressBar

class FeatureExtractorBase(RyanOnTheInside):
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

class AudioFeatureExtractor(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "feature_pipe": ("FEATURE_PIPE",),
                "feature_type": (["amplitude_envelope",  "spectral_centroid", "onset_detection", "chroma_features"],),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "extract_feature"
    

    def extract_feature(self, audio, feature_pipe, feature_type):
        feature = AudioFeature(feature_type, audio, feature_pipe.frame_count, feature_pipe.frame_rate, feature_type)
        feature.extract()
        return (feature, feature_pipe)

class FirstFeature(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "create_feature"
   
class TimeFeatureNode(FirstFeature):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "effect_type": (["smooth", "accelerate", "pulse", "sawtooth", "bounce"],),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

   

    def create_feature(self, effect_type, speed, offset, video_frames, frame_rate):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        time_feature = TimeFeature("time_effect", feature_pipe.frame_rate, feature_pipe.frame_count, 
                                   effect_type=effect_type, speed=speed, offset=offset)
        time_feature.extract()
        return (time_feature, feature_pipe)
    
class DepthFeatureNode(FirstFeature):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "depth_maps": ("IMAGE",),
                "feature_type": (["mean_depth", "depth_variance", "depth_range", "gradient_magnitude", "foreground_ratio", "midground_ratio", "background_ratio"],),
            }
        }

    def create_feature(self, depth_maps, frame_rate, video_frames, feature_type):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        depth_feature = DepthFeature("depth_feature", feature_pipe.frame_rate, feature_pipe.frame_count, depth_maps, feature_type)
        depth_feature.extract()
        return (depth_feature, feature_pipe)

class ColorFeatureNode(FirstFeature):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "feature_type": (["dominant_color", "color_variance", "saturation", "red_ratio", "green_ratio", "blue_ratio"],),
            }
        }

    def create_feature(self, video_frames, frame_rate, feature_type):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        color_feature = ColorFeature("color_feature", feature_pipe.frame_rate, feature_pipe.frame_count, video_frames, feature_type)
        color_feature.extract()
        return (color_feature, feature_pipe)

class BrightnessFeatureNode(FirstFeature):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "feature_type": (["mean_brightness", "brightness_variance", "dark_ratio", "mid_ratio", "bright_ratio"],),
            }
        }

    def create_feature(self, video_frames, frame_rate, feature_type):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        brightness_feature = BrightnessFeature("brightness_feature", feature_pipe.frame_rate, feature_pipe.frame_count, video_frames, feature_type)
        brightness_feature.extract()
        return (brightness_feature, feature_pipe)
    
class MotionFeatureNode(FirstFeature):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "feature_type": (["mean_motion", "max_motion", "motion_direction", "horizontal_motion", "vertical_motion", "motion_complexity","motion_speed"],),
                "flow_method": (["Farneback", "LucasKanade", "PyramidalLK"],),
                "flow_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "magnitude_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "create_feature"

    def create_feature(self, video_frames, frame_rate, feature_type, flow_method, flow_threshold, magnitude_threshold):
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
            feature_type,
            flow_method,
            flow_threshold,
            magnitude_threshold,
            progress_callback=progress_callback
        )

        motion_feature.extract()
        self.end_progress()

        return (motion_feature, feature_pipe)