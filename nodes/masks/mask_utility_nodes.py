import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import matplotlib.font_manager
import textwrap
import cv2
from tqdm import tqdm
import comfy.utils
from scipy import interpolate
from ...tooltips import apply_tooltips


@apply_tooltips
class MovingShape:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_width": ("INT", {"default": 512, "min": 1, "max": 3840, "step": 1}),
                "frame_height": ("INT", {"default": 512, "min": 1, "max": 2160, "step": 1}),
                "num_frames": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "rgb": ("STRING", {"default": "(255,255,255)"}),
                "shape": (["square", "circle", "triangle"],),
                "shape_width_percent": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.1}),
                "shape_height_percent": ("FLOAT", {"default": 50, "min": 0, "max": 100, "step": 0.1}),
                "shape_start_position_x": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "shape_start_position_y": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "shape_end_position_x": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "shape_end_position_y": ("FLOAT", {"default": 0, "min": -100, "max": 100, "step": 0.1}),
                "movement_type": (["linear", "ease_in_out", "bounce", "elastic"],),
                "grow": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.1}),
                "palindrome": ("BOOLEAN", {"default": False}),
                "delay": ("INT", {"default": 0, "min": 0, "max": 60, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate"
    CATEGORY = "RyanOnTheInside/masks/"

    def generate(self, frame_width, frame_height, num_frames, rgb, shape, shape_width_percent, shape_height_percent, shape_start_position_x, shape_start_position_y, shape_end_position_x, shape_end_position_y, movement_type, grow, palindrome, delay):
        rgb = self.parse_rgb_string(rgb)
        
        # Calculate initial shape dimensions
        initial_width = int(frame_width * shape_width_percent / 100)
        initial_height = int(frame_height * shape_height_percent / 100)

        # Calculate start and end positions
        def calculate_position(pos_x, pos_y):
            if pos_x == 0:
                x = frame_width // 2 - initial_width // 2
            elif pos_x < 0:
                x = int((pos_x + 100) / 200 * (frame_width + initial_width)) - initial_width
            else:
                x = int(pos_x / 200 * (frame_width + initial_width))

            if pos_y == 0:
                y = frame_height // 2 - initial_height // 2
            elif pos_y < 0:
                y = int((pos_y + 100) / 200 * (frame_height + initial_height)) - initial_height
            else:
                y = int(pos_y / 200 * (frame_height + initial_height))

            return x, y

        start_x, start_y = calculate_position(shape_start_position_x, shape_start_position_y)
        end_x, end_y = calculate_position(shape_end_position_x, shape_end_position_y)

        # Adjust number of frames for palindrome and delay
        if palindrome:
            actual_frames = (num_frames - delay) // 2 + 1
        else:
            actual_frames = num_frames - delay

        # Create interpolation function
        t = np.linspace(0, 1, actual_frames)
        if movement_type == 'linear':
            interp_func = interpolate.interp1d([0, 1], [0, 1])
        elif movement_type == 'ease_in_out':
            interp_func = lambda x: x**2 * (3 - 2*x)
        elif movement_type == 'bounce':
            def bounce(x):
                return 1 - (1 - x)**2 * np.sin(x * np.pi * 5)**2
            interp_func = bounce
        elif movement_type == 'elastic':
            def elastic(x):
                return x**2 * np.sin(10 * np.pi * x)
            interp_func = elastic
        else:
            # Default to linear if an invalid movement_type is provided
            interp_func = lambda x: x

        progress = interp_func(t)
        if palindrome:
            progress = np.concatenate([progress, progress[-2:0:-1]])

        # Add delay frames
        progress = np.concatenate([np.zeros(delay), progress])

        images = []
        pbar = comfy.utils.ProgressBar(num_frames)

        for i, prog in tqdm(enumerate(progress), desc='Generating frames', total=num_frames):
            mask = np.zeros((frame_height, frame_width, 1), dtype=np.float32)

            # Calculate shape position
            pos_x = int(start_x + (end_x - start_x) * prog)
            pos_y = int(start_y + (end_y - start_y) * prog)

            # Apply growth
            max_width = frame_width
            max_height = frame_height
            current_width = int(initial_width + (max_width - initial_width) * prog * grow / 100)
            current_height = int(initial_height + (max_height - initial_height) * prog * grow / 100)

            # Calculate the center of the shape
            center_x = pos_x + current_width // 2
            center_y = pos_y + current_height // 2

            if shape == "square":
                cv2.rectangle(mask, (pos_x, pos_y), (pos_x + current_width, pos_y + current_height), 1, -1)
            elif shape == "circle":
                radius = min(current_width, current_height) // 2
                cv2.circle(mask, (center_x, center_y), radius, 1, -1)
            elif shape == "triangle":
                points = np.array([
                    [center_x, pos_y],
                    [pos_x, pos_y + current_height],
                    [pos_x + current_width, pos_y + current_height]
                ], np.int32)
                cv2.fillPoly(mask, [points], 1)

            images.append(torch.from_numpy(mask))
            pbar.update(1)

        images_tensor = torch.stack(images, dim=0).squeeze(-1)
        return (images_tensor,)

    def parse_rgb_string(self, rgb_string):
        try:
            components = rgb_string.strip('()').split(',')
            if len(components) != 3:
                raise ValueError
            rgb = tuple(int(component) for component in components)
            if not all(0 <= component <= 255 for component in rgb):
                raise ValueError
            return rgb
        except ValueError:
            raise ValueError("Invalid RGB!")
        
@apply_tooltips
class TextMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        font_list = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "text": ("STRING", {"default": "Hello World"}),
                "font": (font_list,),
                "font_size": ("INT", {"default": 32, "min": 1, "max": 1000}),
                "font_color": ("STRING", {"default": "(255,255,255)"}),
                "background_color": ("STRING", {"default": "(0,0,0)"}),
                "x_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.01}),
                "y_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.01}),
                "rotation": ("FLOAT", {"default": 0, "min": 0, "max": 360}),
                "max_width_ratio": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "create_text_mask"

    CATEGORY = "RyanOnTheInside/masks/"

    def parse_rgb(self, rgb_string):
        try:
            r, g, b = map(int, rgb_string.strip('()').split(','))
            return (r, g, b)
        except:
            print(f"Failed to parse RGB string: {rgb_string}. Using default.")
            return (0, 0, 0)  # Default to black if parsing fails

    def create_text_mask(self, width, height, text, font, font_size, font_color, background_color, x_position, y_position, rotation, max_width_ratio, batch_size):
        font_color = self.parse_rgb(font_color)
        background_color = self.parse_rgb(background_color)

        masks = []
        images = []

        try:
            font_path = matplotlib.font_manager.findfont(matplotlib.font_manager.FontProperties(family=font))
            font_obj = ImageFont.truetype(font_path, font_size)
        except:
            print(f"Failed to load font {font}. Using default font.")
            font_obj = ImageFont.load_default()

        max_width = int(width * max_width_ratio)

        wrapped_text = textwrap.fill(text, width=max_width // font_size)

        temp_img = Image.new('RGB', (width, height))
        temp_draw = ImageDraw.Draw(temp_img)

        text_bbox = temp_draw.multiline_textbbox((0, 0), wrapped_text, font=font_obj)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x = int((width - text_width) * x_position)
        y = int((height - text_height) * y_position)

        for _ in range(batch_size):
            image = Image.new('RGB', (width, height), color=background_color)
            draw = ImageDraw.Draw(image)

            txt_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)

            txt_draw.multiline_text((x, y), wrapped_text, font=font_obj, fill=font_color)

            rotated_txt_img = txt_img.rotate(rotation, expand=1, fillcolor=(0, 0, 0, 0))

            image.paste(rotated_txt_img, (0, 0), rotated_txt_img)

            gray_image = image.convert('L')

            mask = np.array(gray_image).astype(np.float32) / 255.0
            masks.append(mask)

            image_np = np.array(image).astype(np.float32) / 255.0
            images.append(image_np)

        mask_tensor = torch.from_numpy(np.stack(masks))
        image_tensor = torch.from_numpy(np.stack(images))

        return (mask_tensor, image_tensor)

@apply_tooltips
class _mfc:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "red": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "green": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "blue": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "threshold": ("INT", { "default": 0, "min": 0, "max": 127, "step": 1, }),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "execute"
    CATEGORY = "RyanOnTheInside/masks/"

    def execute(self, image, red, green, blue, threshold):
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        color = torch.tensor([red, green, blue], device=image.device)
        lower_bound = (color - threshold).clamp(min=0)
        upper_bound = (color + threshold).clamp(max=255)
        lower_bound = lower_bound.view(1, 1, 1, 3)
        upper_bound = upper_bound.view(1, 1, 1, 3)
        mask = (temp >= lower_bound) & (temp <= upper_bound)
        mask = mask.all(dim=-1)
        mask = mask.float()
        
        # Create an image of the mask
        mask_image = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        
        return (mask, mask_image)
    

@apply_tooltips
class MaskCompositePlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
                "operation": (["add", "subtract", "multiply", "divide", "min", "max", "pixel_wise_min", "pixel_wise_max"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "composite_masks"
    CATEGORY = "RyanOnTheInside/masks/"

    def composite_masks(self, mask1, mask2, operation):
        # Ensure masks have the same shape
        if mask1.shape != mask2.shape:
            raise ValueError("Masks must have the same shape")

        if operation == "add":
            result = torch.clamp(mask1 + mask2, 0, 1)
        elif operation == "subtract":
            result = torch.clamp(mask1 - mask2, 0, 1)
        elif operation == "multiply":
            result = mask1 * mask2
        elif operation == "divide":
            result = torch.where(mask2 != 0, mask1 / mask2, mask1)
        elif operation == "min":
            result = torch.min(mask1, mask2)
        elif operation == "max":
            result = torch.max(mask1, mask2)
        elif operation == "pixel_wise_min":
            result = torch.where(mask1 < mask2, mask1, mask2)
        elif operation == "pixel_wise_max":
            result = torch.where(mask1 > mask2, mask1, mask2)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return (result,)

@apply_tooltips
class AdvancedLuminanceMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "luminance_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glow_radius": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
                "edge_preservation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "background_samples": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "denoise_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "create_mask"
    CATEGORY = "RyanOnTheInside/masks/"

    def create_mask(self, image, luminance_threshold, glow_radius, edge_preservation, background_samples, denoise_strength):
        # Convert to numpy for OpenCV operations
        device = image.device
        image_np = image.cpu().numpy()
        
        # Get batch size and prepare output arrays
        batch_size = image_np.shape[0] if len(image_np.shape) == 4 else 1
        if batch_size == 1 and len(image_np.shape) == 3:
            image_np = image_np[None, ...]
        
        masks = []
        vis_images = []
        
        # Process each frame in the batch
        for b in range(batch_size):
            frame = image_np[b]
            
            # Convert to various color spaces for analysis
            frame_rgb = (frame * 255).astype(np.uint8)
            frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
            
            # Estimate background color from corners
            h, w = frame_gray.shape
            corner_size = max(h, w) // background_samples
            corners = [
                frame_gray[:corner_size, :corner_size],
                frame_gray[:corner_size, -corner_size:],
                frame_gray[-corner_size:, :corner_size],
                frame_gray[-corner_size:, -corner_size:]
            ]
            bg_value = np.median([np.median(corner) for corner in corners])
            
            # Create initial mask based on luminance difference from background
            diff_from_bg = np.abs(frame_gray.astype(float) - bg_value)
            initial_mask = diff_from_bg > (luminance_threshold * 255)
            
            # Apply bilateral filter to reduce noise while preserving edges
            if denoise_strength > 0:
                d = int(denoise_strength * 10)
                sigma_color = denoise_strength * 75
                sigma_space = denoise_strength * 75
                initial_mask = cv2.bilateralFilter(initial_mask.astype(np.float32), d, sigma_color, sigma_space)
            
            # Enhance edges and glowing effects
            if glow_radius > 0:
                kernel_size = 2 * glow_radius + 1
                blurred = cv2.GaussianBlur(initial_mask, (kernel_size, kernel_size), 0)
                mask = np.maximum(initial_mask, blurred * edge_preservation)
            else:
                mask = initial_mask
                
            # Calculate luminance-based alpha for transparency
            luminance = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)[:,:,0] / 255.0
            alpha = np.clip(luminance * (1.0 / luminance_threshold), 0, 1)
            
            # Combine mask with alpha
            final_mask = mask * alpha
            
            masks.append(final_mask)
            vis_images.append(np.stack([final_mask, final_mask, final_mask], axis=-1))
        
        # Convert back to torch tensors
        masks_tensor = torch.from_numpy(np.stack(masks)).float().to(device)
        vis_tensor = torch.from_numpy(np.stack(vis_images)).float().to(device)
        
        return (masks_tensor, vis_tensor)

@apply_tooltips
class TranslucentComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background": ("IMAGE",),
                "foreground": ("IMAGE",),
                "mask": ("MASK",),
                "blend_mode": (["normal", "screen", "multiply", "overlay"],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preserve_transparency": ("BOOLEAN", {"default": True}),
                "luminance_boost": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "background_influence": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "RyanOnTheInside/masks/"

    def composite(self, background, foreground, mask, blend_mode, opacity, preserve_transparency, luminance_boost, background_influence):
        device = background.device
        
        # Convert to numpy for processing
        bg = background.cpu().numpy()
        fg = foreground.cpu().numpy()
        mask = mask.cpu().numpy()
        
        # Handle batching
        if len(bg.shape) == 3:
            bg = bg[None, ...]
        if len(fg.shape) == 3:
            fg = fg[None, ...]
        if len(mask.shape) == 2:
            mask = mask[None, ...]
            
        batch_size = bg.shape[0]
        result = []
        
        for b in range(batch_size):
            # Convert to float32 for processing
            bg_frame = (bg[b] * 255).astype(np.float32)
            fg_frame = (fg[b] * 255).astype(np.float32)
            mask_frame = mask[b]
            
            # Convert to LAB color space for luminance processing
            bg_lab = cv2.cvtColor(bg_frame.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            fg_lab = cv2.cvtColor(fg_frame.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # Apply luminance boost to foreground
            if luminance_boost != 0:
                fg_lab[:, :, 0] = np.clip(fg_lab[:, :, 0] + (luminance_boost * 100), 0, 255)
                fg_frame = cv2.cvtColor(fg_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
            
            # Calculate transparency based on luminance if preserve_transparency is True
            if preserve_transparency:
                fg_luminance = cv2.cvtColor(fg_frame.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                mask_frame = mask_frame * fg_luminance
            
            # Apply blend mode
            if blend_mode == "screen":
                blended = 255 - ((255 - bg_frame) * (255 - fg_frame) / 255)
            elif blend_mode == "multiply":
                blended = (bg_frame * fg_frame) / 255
            elif blend_mode == "overlay":
                blended = np.where(bg_frame > 127.5,
                                 255 - ((255 - 2*(bg_frame-127.5)) * (255-fg_frame)) / 255,
                                 (2*bg_frame*fg_frame) / 255)
            else:  # normal
                blended = fg_frame
            
            # Apply background influence
            if background_influence > 0:
                bg_influence_mask = cv2.GaussianBlur(mask_frame[..., None], (5, 5), 0) * background_influence
                blended = blended * (1 - bg_influence_mask) + bg_frame * bg_influence_mask
            
            # Final compositing with mask and opacity
            mask_3ch = np.stack([mask_frame] * 3, axis=-1) * opacity
            composite = bg_frame * (1 - mask_3ch) + blended * mask_3ch
            
            # Normalize and append to results
            result.append(np.clip(composite, 0, 255) / 255.0)
        
        # Convert back to torch tensor
        result_tensor = torch.from_numpy(np.stack(result)).float().to(device)
        
        return (result_tensor,)