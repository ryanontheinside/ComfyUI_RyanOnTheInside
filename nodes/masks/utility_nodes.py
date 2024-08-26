import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import matplotlib.font_manager
import textwrap
import cv2
from tqdm import tqdm
import comfy.utils
from scipy import interpolate


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
    CATEGORY = "/RyanOnTheInside/masks/"

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

    CATEGORY = "/RyanOnTheInside/masks/"

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
    #TODO add image
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "/RyanOnTheInside/masks/"

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
        return (mask, )