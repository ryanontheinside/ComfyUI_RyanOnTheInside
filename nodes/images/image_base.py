# import torch
# import numpy as np
# from abc import ABC, abstractmethod
# from tqdm import tqdm
# from comfy.utils import ProgressBar
# from .image_utils import preserve_original_colors, apply_blur
# from ... import RyanOnTheInside

# class ImageBase(RyanOnTheInside, ABC):
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "images": ("IMAGE",),
#                 "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light"],),
#                 "preserve_colors": ("BOOLEAN", {"default": False}),
#                 "apply_to_alpha": ("BOOLEAN", {"default": False}),
#             }
#         }

#     CATEGORY = "RyanOnTheInside/Images"

#     def __init__(self):
#         self.pre_processors = []
#         self.post_processors = []
#         self.progress_bar = None
#         self.tqdm_bar = None
#         self.current_progress = 0
#         self.total_steps = 0

#     def add_pre_processor(self, func):
#         self.pre_processors.append(func)
#         return self

#     def add_post_processor(self, func):
#         self.post_processors.append(func)
#         return self

#     def pre_process(self, image):
#         for processor in self.pre_processors:
#             image = processor(image)
#         return image

#     def post_process(self, image):
#         for processor in self.post_processors:
#             image = processor(image)
#         return image

#     def start_progress(self, total_steps, desc="Processing"):
#         self.progress_bar = ProgressBar(total_steps)
#         self.tqdm_bar = tqdm(total=total_steps, desc=desc, leave=False)
#         self.current_progress = 0
#         self.total_steps = total_steps

#     def update_progress(self, step=1):
#         self.current_progress += step
#         if self.progress_bar:
#             self.progress_bar.update(step)
#         if self.tqdm_bar:
#             self.tqdm_bar.update(step)

#     def end_progress(self):
#         if self.tqdm_bar:
#             self.tqdm_bar.close()
#         self.progress_bar = None
#         self.tqdm_bar = None
#         self.current_progress = 0
#         self.total_steps = 0

#     @abstractmethod
#     def process_image(self, image: np.ndarray, strength: float, **kwargs) -> np.ndarray:
#         """
#         Process a single image. This method must be implemented by child classes.
#         """
#         pass

#     def apply_image_operation(self, processed_images: torch.Tensor, original_images: torch.Tensor, strength: float, blend_mode: str, preserve_colors: bool, apply_to_alpha: bool, **kwargs) -> torch.Tensor:
#         processed_images_np = processed_images.cpu().numpy() if isinstance(processed_images, torch.Tensor) else processed_images
#         original_images_np = original_images.cpu().numpy() if isinstance(original_images, torch.Tensor) else original_images
#         num_frames = processed_images_np.shape[0]

#         self.start_progress(num_frames, desc="Applying image operation")

#         result = []
#         for processed_image, original_image in zip(processed_images_np, original_images_np):
#             # Pre-processing
#             processed_image = self.pre_process(processed_image)

#             if preserve_colors:
#                 processed_image = preserve_original_colors(original_image, processed_image)

#             if not apply_to_alpha and original_image.shape[-1] == 4:
#                 processed_image[..., -1] = original_image[..., -1]

#             # Post-processing
#             processed_image = self.post_process(processed_image)

#             # Ensure the final image is clipped between 0 and 1
#             processed_image = np.clip(processed_image, 0, 1)

#             result.append(processed_image)
#             self.update_progress()

#         self.end_progress()

#         return torch.from_numpy(np.stack(result)).float()

#     @abstractmethod
#     def main_function(self, *args, **kwargs) -> torch.Tensor:
#         """
#         Main entry point for the node. This method must be implemented by child classes.
#         It should call apply_image_operation with the appropriate arguments.
#         """
#         pass



# import torch
# import numpy as np
# from abc import abstractmethod
# from .image_base import ImageBase

# class FlexImageBase(ImageBase):

#     BLEND_MODES = ["multiply", "add", "subtract", "override"]
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 **super().INPUT_TYPES()["required"],
#                 "feature": ("FEATURE",),
#                 "feature_pipe": ("FEATURE_PIPE",),
#                 "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "feature_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
#                 "feature_blend_mode": (cls.BLEND_MODES,),
#             }
#         }

#     CATEGORY = "RyanOnTheInside/FlexImages"
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "main_function"

#     @abstractmethod
#     def process_image(self, image: np.ndarray, feature_value: float, strength: float, **kwargs) -> np.ndarray:
#         pass

#     def apply_image_operation(self, images, feature, feature_pipe, strength, blend_mode, preserve_colors, apply_to_alpha, feature_threshold, feature_strength, feature_blend_mode, **kwargs):
#         num_frames = feature_pipe.frame_count
#         original_images = images.clone()

#         self.start_progress(num_frames, desc="Applying flex image operation")

#         result = []
#         for i in range(num_frames):
#             image = images[i].numpy()
#             feature_value = feature.get_value_at_frame(i)
            
#             if feature_value >= feature_threshold:
#                 # Apply the effect with the feature value directly controlling the intensity
#                 processed_image = self.process_image(image, feature_value * feature_strength, strength, **kwargs)
#             else:
#                 processed_image = image

#             result.append(processed_image)
#             self.update_progress()

#         self.end_progress()

#         processed_images = torch.from_numpy(np.stack(result)).float()
#         return super().apply_image_operation(processed_images, original_images, strength, blend_mode, preserve_colors, apply_to_alpha, **kwargs)

#     @abstractmethod
#     def main_function(self, images, feature, feature_pipe, strength, blend_mode, preserve_colors, apply_to_alpha, feature_threshold, feature_strength, feature_blend_mode, **kwargs):
#         pass



import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
from ... import RyanOnTheInside

class FlexImageBase(RyanOnTheInside, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "feature": ("FEATURE",),
                "feature_pipe": ("FEATURE_PIPE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "modulate_param": (cls.get_modifiable_params(), {"default": cls.get_modifiable_params()[0]}),
                "modulation_mode": (["relative", "absolute"], {"default": "relative"}),
            }
        }

    CATEGORY = "RyanOnTheInside/ImageEffects"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return []

    def modulate_param(self, param_name, param_value, feature_value, strength, mode):
        if mode == "relative":
            return param_value * (1 + (feature_value - 0.5) * strength)
        else:  # absolute
            return param_value * feature_value * strength

    def apply_effect(self, images, feature, feature_pipe, strength, feature_threshold, **kwargs):
        num_frames = feature_pipe.frame_count
        images_np = images.cpu().numpy()

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            image = images_np[i]
            feature_value = feature.get_value_at_frame(i)
            
            if feature_value >= feature_threshold:
                processed_image = self.process_image(image, feature_value, strength, **kwargs)
            else:
                processed_image = image

            result.append(processed_image)
            self.update_progress()

        self.end_progress()

        # Convert the list of numpy arrays to a single numpy array
        result_np = np.stack(result)
        
        # Convert the numpy array to a PyTorch tensor and ensure it's in BHWC format
        result_tensor = torch.from_numpy(result_np).float()
        
        # If the tensor is not in BHWC format, transpose it
        if result_tensor.shape[1] == 3:  # If it's in BCHW format
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    def process_image(self, image: np.ndarray, feature_value: float, strength: float, 
                      modulate_param: str, modulation_mode: str, **kwargs) -> np.ndarray:
        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                if param_name == modulate_param:
                    kwargs[param_name] = self.modulate_param(param_name, kwargs[param_name], 
                                                             feature_value, strength, modulation_mode)
        
        # Call the child class's implementation
        return self.apply_effect_internal(image, **kwargs)

    @abstractmethod
    def apply_effect_internal(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the image. To be implemented by child classes."""
        pass