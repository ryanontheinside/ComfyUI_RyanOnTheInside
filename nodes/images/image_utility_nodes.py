import torch
import torch.nn.functional as F
from ..node_utilities import string_to_rgb 
from ... import RyanOnTheInside
from comfy.utils import ProgressBar, common_upscale
from ... import ProgressMixin
from ...tooltips import apply_tooltips

class ImageUtilityNode(ProgressMixin):
    CATEGORY = "RyanOnTheInside/Utility/Images"

@apply_tooltips
class DyeImage(ImageUtilityNode):
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

    def dye_image(self, image, source_rgb, target_rgb, tolerance):
        
        source = torch.tensor(string_to_rgb(source_rgb), dtype=torch.float32, device=image.device)
        target = torch.tensor(string_to_rgb(target_rgb), dtype=torch.float32, device=image.device)

        color_distance = torch.sum((image - source.view(1, 1, 1, 3)).abs(), dim=-1)
        mask = color_distance <= (tolerance * 3)   

        result = image.clone()
        result[mask] = target.view(1, 1, 1, 3).expand_as(result)[mask]

        return (result,)



# From https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
# THEN from comfyui essentials shoutout MATEO
@apply_tooltips
class ImageCASBatch(ImageUtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    def execute(self, image, amount, batch_size):
        # Process images in batches
        output_batches = []
        self.start_progress(image.shape[0] // batch_size)
        for i in range(0, image.shape[0], batch_size):
            batch = image[i:i+batch_size]
            output_batch = self.process_batch(batch, amount)
            output_batches.append(output_batch)
            self.update_progress()
        self.end_progress()
        # Concatenate all processed batches
        output = torch.cat(output_batches, dim=0)
        return (output,)

    def process_batch(self, batch, amount):
        epsilon = 1e-5
        img = F.pad(batch.permute([0,3,1,2]), pad=(1, 1, 1, 1))

        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]

        # Computing contrast
        cross = (b, d, e, f, h)
        mn = torch.min(torch.stack(cross), dim=0).values
        mx = torch.max(torch.stack(cross), dim=0).values

        diag = (a, c, g, i)
        mn2 = torch.min(torch.stack(diag), dim=0).values
        mx2 = torch.max(torch.stack(diag), dim=0).values
        mx = mx + mx2
        mn = mn + mn2

        # Computing local weight
        inv_mx = torch.reciprocal(mx + epsilon)
        amp = inv_mx * torch.minimum(mn, (2 - mx))

        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        output = ((b + d + f + h)*w + e) * div
        output = output.clamp(0, 1)
        #output = torch.nan_to_num(output)

        output = output.permute([0,2,3,1])

        return output


@apply_tooltips
class ImageScaleToTarget(ImageUtilityNode):
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), 
                             "target_image": ("IMAGE",), 
                             "upscale_method": (s.upscale_methods,),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    def upscale(self, image, upscale_method, target_image, crop):
        b,height,width,c = target_image.shape
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = common_upscale(samples, width, height, upscale_method, crop)
            s = s.movedim(1,-1)
        return (s,)
    

import json

class Color_Picker:
    """
    A node that provides a color picker interface and outputs hex color, RGB color, and hue values.
    """
    
    CATEGORY = "RyanOnTheInside"
    FUNCTION = "process_color"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("hex_color", "rgb_color", "hue_shift")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("STRING", {"default": '{"hex":"#FF0000","rgb":"255,0,0","hue":0}'})
            }
        }

    def process_color(self, color):
        try:
            # Parse the JSON data from the widget
            color_data = json.loads(color)
            
            # Extract values with defaults
            hex_color = color_data.get("hex", "#FF0000").upper()  # Ensure uppercase for consistency
            rgb_value = color_data.get("rgb", "255,0,0")
            hue = color_data.get("hue", 0)
            
            # Convert RGB string to integers for validation
            try:
                r, g, b = map(int, rgb_value.split(','))
                # Ensure RGB values are in valid range
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                # Reconstruct validated RGB string
                rgb_value = f"{r},{g},{b}"
            except ValueError:
                rgb_value = "255,0,0"  # Default if parsing fails
            
            # Ensure hue is in valid range (0-360)
            hue = max(0, min(360, int(hue)))
            
            return (hex_color, rgb_value, hue)
            
        except (json.JSONDecodeError, KeyError):
            # Return defaults if JSON parsing fails
            return ("#FF0000", "255,0,0", 0) 