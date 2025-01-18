import torch
import torch.nn.functional as F
from ..node_utilities import string_to_rgb 
from ... import RyanOnTheInside
from comfy.utils import ProgressBar, common_upscale

from ...tooltips import apply_tooltips

class ImageUtilityNode(RyanOnTheInside):
    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None
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
    

@apply_tooltips
class ColorPicker(ImageUtilityNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},  # No visible inputs needed
            "hidden": {
                "color": ("STRING", {"default": "#FF0000"}),
                "rgb_value": ("STRING", {"default": "255,0,0"}),
                "hue": ("INT", {"default": 0, "min": 0, "max": 360}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "hex_color",      # #FF0000 format
        "rgb_color",      # 255,0,0 format
        "hue_shift",      # 0-360 integer
        "rgb_float",      # 1.0,0.0,0.0 format
        "rgb_percent",    # 100%,0%,0% format
        "hsl_color",      # hsl(0,100%,50%) format
        "hsv_color",      # hsv(0,100%,100%) format
    )
    FUNCTION = "process_color"
    CATEGORY = "RyanOnTheInside/Utility/Images"
    
    def process_color(self, color="#FF0000", rgb_value="255,0,0", hue=0):
        # Convert RGB string to integers
        r, g, b = map(int, rgb_value.split(','))
        
        # Calculate RGB float values (0-1)
        rgb_float = f"{r/255:.3f},{g/255:.3f},{b/255:.3f}"
        
        # Calculate RGB percentage values
        rgb_percent = f"{int(r/255*100)}%,{int(g/255*100)}%,{int(b/255*100)}%"
        
        # Calculate HSL
        r_norm, g_norm, b_norm = r/255, g/255, b/255
        cmax = max(r_norm, g_norm, b_norm)
        cmin = min(r_norm, g_norm, b_norm)
        delta = cmax - cmin
        
        # Calculate lightness
        lightness = (cmax + cmin) / 2
        
        # Calculate saturation
        saturation = 0 if delta == 0 else (delta / (1 - abs(2 * lightness - 1)))
        
        # Format HSL string
        hsl_color = f"hsl({int(hue)},{int(saturation*100)}%,{int(lightness*100)}%)"
        
        # Calculate HSV
        v = cmax
        s = 0 if cmax == 0 else delta/cmax
        
        # Format HSV string
        hsv_color = f"hsv({int(hue)},{int(s*100)}%,{int(v*100)}%)"
        
        return (
            color,          # hex_color
            rgb_value,      # rgb_color
            int(hue),       # hue_shift
            rgb_float,      # rgb_float
            rgb_percent,    # rgb_percent
            hsl_color,      # hsl_color
            hsv_color,      # hsv_color
        )
    