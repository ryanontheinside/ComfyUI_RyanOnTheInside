class FeaturePipe:
    def __init__(self, frame_rate, video_frames):
        self.frame_rate = frame_rate
        self.frame_count = video_frames.shape[0]
        self.height = video_frames.shape[1]
        self.width = video_frames.shape[2]

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
import torch
from ... import RyanOnTheInside

class ManualFeaturePipe(RyanOnTheInside):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0}),
                "frame_count": ("INT", {"default": 100, "min": 1}),
                "width": ("INT", {"default": 1920, "min": 1}),
                "height": ("INT", {"default": 1080, "min": 1}),
            }
        }

    RETURN_TYPES = ("FEATURE_PIPE",)
    FUNCTION = "create_feature_pipe"
    DESCRIPTION = "Create a feature pipe with the specified frame rate, frame count, width, and height. Probably want to use this with manual feature."

    def create_feature_pipe(self, frame_rate, frame_count, width, height):
        # Create a batch of empty tensors for the feature pipe in BHWC format
        video_frames = torch.zeros((frame_count, height, width, 3), dtype=torch.float32)  # Assuming 3 channels for RGB

        feature_pipe = FeaturePipe(frame_rate, video_frames)
        
        return (feature_pipe,)