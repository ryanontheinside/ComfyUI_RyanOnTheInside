import torch
import numpy as np
from ..node_utilities import string_to_rgb 

class DyeImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "source_rgb": ("STRING", {"default": "255,255,255"}),
                "target_rgb": ("STRING", {"default": "0,0,0"}),
                "tolerance": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dye_image"
    CATEGORY = "image/color"

    def dye_image(self, image, source_rgb, target_rgb, tolerance):
        
        source = torch.tensor(string_to_rgb(source_rgb), dtype=torch.float32, device=image.device)
        target = torch.tensor(string_to_rgb(target_rgb), dtype=torch.float32, device=image.device)

        color_distance = torch.sum((image - source.view(1, 1, 1, 3)).abs(), dim=-1)
        mask = color_distance <= (tolerance * 3)   

        result = image.clone()
        result[mask] = target.view(1, 1, 1, 3).expand_as(result)[mask]

        return (result,)
    

class ColorPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("COLOR", {"default": "#FF0000"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "pick_color"
    CATEGORY = "image/color"

    def pick_color(self, color):
        # Convert hex color to RGB
        r, g, b = tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4))
        
        # Create a 1x1x3 image tensor with the selected color
        color_tensor = torch.tensor([[[r, g, b]]], dtype=torch.float32) / 255.0
        
        # Create the RGB string
        rgb_string = f"{r},{g},{b}"
        
        return (color_tensor, rgb_string)