import cv2
import torch
import numpy as np
from .flex_image_base import FlexImageBase
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from .image_utils import transform_image
from ...tooltips import apply_tooltips

@apply_tooltips
class FlexImageEdgeDetect(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "low_threshold": ("FLOAT", {"default": 100, "min": 0, "max": 255, "step": 1}),
            "high_threshold": ("FLOAT", {"default": 200, "min": 0, "max": 255, "step": 1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["low_threshold", "high_threshold", "None"]

    def apply_effect_internal(self, image: np.ndarray, low_threshold: float, high_threshold: float, **kwargs) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(np.uint8(gray * 255), low_threshold, high_threshold)
        
        # Convert back to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return edges_rgb.astype(float) / 255.0

@apply_tooltips
class FlexImagePosterize(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "max_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
            "dither_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "channel_separation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "gamma": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 2.2, "step": 0.1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["max_levels", "dither_strength", "channel_separation", "gamma", "None"]

    def apply_effect_internal(self, image: np.ndarray, max_levels: int, dither_strength: float, 
                              channel_separation: float, gamma: float, **kwargs) -> np.ndarray:
        # Apply gamma correction
        image_gamma = np.power(image, gamma)
        
        # Convert image to uint8
        image_uint8 = (image_gamma * 255).astype(np.uint8)
        
        posterized = np.zeros_like(image_uint8)
        
        for c in range(3):  # RGB channels
            # Calculate levels for each channel
            channel_levels = int(np.clip(2 + (max_levels - 2) * (1 + channel_separation * (c - 1)), 2, max_levels))
            
            # Posterize
            div = 256 // channel_levels
            posterized[:,:,c] = (image_uint8[:,:,c] // div) * div
        
        if dither_strength > 0:
            # Apply Floyd-Steinberg dithering with adjustable strength
            error = (image_uint8.astype(np.float32) - posterized) * dither_strength
            h, w = image.shape[:2]
            for c in range(3):  # RGB channels
                for i in range(h - 1):
                    for j in range(w - 1):
                        old_pixel = posterized[i, j, c]
                        new_pixel = np.clip(old_pixel + error[i, j, c], 0, 255)
                        quant_error = old_pixel - new_pixel
                        posterized[i, j, c] = new_pixel
                        error[i, j+1, c] += quant_error * 7 / 16
                        error[i+1, j-1, c] += quant_error * 3 / 16
                        error[i+1, j, c] += quant_error * 5 / 16
                        error[i+1, j+1, c] += quant_error * 1 / 16
        
        # Convert back to float32 and apply gamma correction
        return np.power(posterized.astype(np.float32) / 255.0, 1/gamma)
    
@apply_tooltips
class FlexImageKaleidoscope(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "segments": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),
            "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "zoom": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            "precession": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["segments", "zoom", "rotation", "precession", "speed", "None"]

    def apply_effect_internal(self, image: np.ndarray, segments: int, center_x: float, center_y: float, 
                              zoom: float, rotation: float, precession: float, speed: float, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        center = (int(w * center_x), int(h * center_y))
        
        # Ensure segments is an integer
        segments = max(2, int(segments))
        
        # Create the kaleidoscope effect
        segment_angle = 360 / segments
        result = np.zeros_like(image)
        
        for i in range(segments):
            angle = i * segment_angle + rotation * speed
            # Apply precession effect
            precession_factor = 1 + precession * (i / segments) * speed
            matrix = cv2.getRotationMatrix2D(center, angle, zoom * precession_factor)
            rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            mask = np.zeros((h, w), dtype=np.float32)
            pts = np.array([center, (0, 0), (w, 0), (w, h), (0, h)], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1.0)
            
            segment_mask = cv2.warpAffine(mask, matrix, (w, h))
            result = np.maximum(result, rotated * segment_mask[:,:,None])
        
        # Ensure the result is not all black
        if np.max(result) == 0:
            print("Warning: Kaleidoscope effect resulted in a black image")
            return image  # Return original image if result is all black
        
        return result.clip(0, 1)
    
@apply_tooltips
class FlexImageColorGrade(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "lut_file": ("STRING", {"default": ""}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["intensity", "mix"]

    def __init__(self):
        super().__init__()
        self.lut_cache = {}

    def load_lut(self, lut_file):
        if lut_file is None or lut_file == "":
            return None
        if lut_file not in self.lut_cache:
            lut = cv2.imread(lut_file, cv2.IMREAD_UNCHANGED)
            if lut is None:
                print(f"Warning: Failed to load LUT file: {lut_file}")
                return None
            self.lut_cache[lut_file] = lut
        return self.lut_cache[lut_file]

    def apply_effect_internal(self, image: np.ndarray, intensity: float, mix: float, 
                              lut_file: str = None, **kwargs) -> np.ndarray:
        # Load the LUT
        lut = self.load_lut(lut_file)

        # Apply color grading if LUT is available
        if lut is not None:
            graded = cv2.LUT(image, lut)
            
            # Apply intensity
            graded = cv2.addWeighted(image, 1 - intensity, graded, intensity, 0)
            
            # Mix with original
            result = cv2.addWeighted(image, 1 - mix, graded, mix, 0)
        else:
            result = image  # If no LUT is available, return the original image

        return np.clip(result, 0, 1)

@apply_tooltips
class FlexImageGlitch(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "shift_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "scan_lines": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
            "color_shift": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["shift_amount", "scan_lines", "color_shift", "None"]

    def apply_effect_internal(self, image: np.ndarray, shift_amount: float, scan_lines: int, color_shift: float, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]

        # Apply horizontal shift
        shift = int(w * shift_amount)
        result = np.roll(image, shift, axis=1)

        # Add scan lines
        if scan_lines > 0:
            scan_line_mask = np.zeros((h, w))
            scan_line_mask[::scan_lines] = 1
            result = result * (1 - scan_line_mask)[:,:,np.newaxis] + scan_line_mask[:,:,np.newaxis]

        # Apply color channel shift
        if color_shift > 0:
            color_shift_amount = int(w * color_shift)
            result[:,:,0] = np.roll(result[:,:,0], color_shift_amount, axis=1)
            result[:,:,2] = np.roll(result[:,:,2], -color_shift_amount, axis=1)

        return np.clip(result, 0, 1)

@apply_tooltips
class FlexImageChromaticAberration(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "shift_amount": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5, "step": 0.001}),
            "angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["shift_amount", "angle", "None"]

    def apply_effect_internal(self, image: np.ndarray, shift_amount: float, angle: float, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]

        # Calculate shift vectors
        dx = int(w * shift_amount * np.cos(np.radians(angle)))
        dy = int(h * shift_amount * np.sin(np.radians(angle)))

        # Shift red and blue channels
        result = np.zeros_like(image)
        result[:,:,0] = np.roll(image[:,:,0], (dy, dx), (0, 1))  # Red channel
        result[:,:,1] = image[:,:,1]  # Green channel (no shift)
        result[:,:,2] = np.roll(image[:,:,2], (-dy, -dx), (0, 1))  # Blue channel

        return np.clip(result, 0, 1)

@apply_tooltips
class FlexImagePixelate(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "pixel_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["pixel_size", "None"]

    def apply_effect_internal(self, image: np.ndarray, pixel_size: int, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]

        # Ensure pixel_size is an integer and greater than 0
        pixel_size = max(1, int(pixel_size))

        # Calculate new dimensions
        new_h, new_w = h // pixel_size, w // pixel_size

        # Ensure new dimensions are at least 1x1
        new_h, new_w = max(1, new_h), max(1, new_w)

        # Resize down
        small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Resize up
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return result
    
@apply_tooltips
class FlexImageBloom(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "blur_amount": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["threshold", "blur_amount", "intensity", "None"]

    def apply_effect_internal(self, image: np.ndarray, threshold: float, blur_amount: float, 
                              intensity: float, **kwargs) -> np.ndarray:
        # Extract bright areas
        bright_areas = np.maximum(image - threshold, 0) / (1 - threshold)

        # Apply gaussian blur
        blurred = gaussian_filter(bright_areas, sigma=blur_amount)

        # Combine with original image
        result = image + blurred * intensity

        return np.clip(result, 0, 1)

@apply_tooltips
class FlexImageTiltShift(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "blur_amount": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            "focus_position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "focus_position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "focus_width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "focus_height": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "focus_shape": (["rectangle", "ellipse"], {"default": "rectangle"}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["blur_amount", "focus_position_x", "focus_position_y", "focus_width", "focus_height", "None"]

    def apply_effect_internal(self, image: np.ndarray, blur_amount: float, focus_position_x: float, 
                              focus_position_y: float, focus_width: float, focus_height: float, 
                              focus_shape: str, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        center_x, center_y = int(w * focus_position_x), int(h * focus_position_y)
        width, height = int(w * focus_width), int(h * focus_height)

        if focus_shape == "rectangle":
            cv2.rectangle(mask, 
                          (center_x - width//2, center_y - height//2),
                          (center_x + width//2, center_y + height//2),
                          1, -1)
        else:  # ellipse
            cv2.ellipse(mask, 
                        (center_x, center_y),
                        (width//2, height//2),
                        0, 0, 360, 1, -1)

        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=min(width, height) / 6)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=blur_amount)
        result = image * mask[:,:,np.newaxis] + blurred * (1 - mask[:,:,np.newaxis])

        return np.clip(result, 0, 1)
    
@apply_tooltips
class FlexImageParallax(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "shift_x": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
            "shift_y": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
            "shift_z": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
        })
        base_inputs["optional"].update({
            "depth_map": ("IMAGE",),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["shift_x", "shift_y", "shift_z"]

    def apply_effect_internal(
        self,
        image: np.ndarray,
        shift_x: float,
        shift_y: float,
        shift_z: float,
        depth_map: np.ndarray = None,  # Default depth_map to None
        frame_index: int = 0,  # Default to frame 0 for consistency
        **kwargs
    ) -> np.ndarray:
        h, w, _ = image.shape

        if depth_map is not None:
            # Depth-based parallax
            depth_map_frame = depth_map[frame_index].cpu().numpy()
            depth_map_gray = np.mean(depth_map_frame, axis=-1)
            depth_map_gray /= np.max(depth_map_gray)

            # Calculate shifts based on the depth map
            dx = (w * shift_x * depth_map_gray).astype(np.int32)
            dy = (h * shift_y * depth_map_gray).astype(np.int32)

            # Scale based on depth map
            scale = 1 + shift_z * depth_map_gray
        else:
            # 2D fallback: no depth map, apply uniform parallax
            dx = int(w * shift_x)
            dy = int(h * shift_y)
            scale = 1 + shift_z

        # Generate the grid for x, y coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply shifts
        x_shifted = x + dx
        y_shifted = y + dy

        # Apply scaling around the center of the image
        cx, cy = w / 2, h / 2
        x_scaled = cx + (x_shifted - cx) * scale
        y_scaled = cy + (y_shifted - cy) * scale

        # Ensure coordinates are within image bounds
        new_x = np.clip(x_scaled, 0, w - 1).astype(np.int32)
        new_y = np.clip(y_scaled, 0, h - 1).astype(np.int32)

        # Generate the resulting image with parallax effect
        result = image[new_y, new_x]

        return np.clip(result, 0, 1)
    
@apply_tooltips
class FlexImageContrast(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
            "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            "preserve_luminosity": ("BOOLEAN", {"default": True}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["contrast", "brightness", "preserve_luminosity", "None"]

    def apply_effect_internal(self, image: np.ndarray, contrast: float, brightness: float, preserve_luminosity: bool, **kwargs) -> np.ndarray:
        # Convert to float32 if not already
        image = image.astype(np.float32)

        # Apply brightness adjustment
        result = image + brightness

        # Apply contrast adjustment
        mean = np.mean(result, axis=(0, 1), keepdims=True)
        result = (result - mean) * contrast + mean

        if preserve_luminosity:
            # Calculate current and original luminosity
            current_luminosity = np.mean(result)
            original_luminosity = np.mean(image)
            
            # Adjust to preserve original luminosity
            result *= original_luminosity / current_luminosity

        return np.clip(result, 0, 1)

import numpy as np
import cv2

@apply_tooltips
class FlexImageWarp(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "warp_type": (["noise", "twist", "bulge"], {"default": "noise"}),
            "warp_strength": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
            "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        })
        base_inputs["optional"].update({
            "warp_frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            "warp_octaves": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
            "warp_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["warp_strength", "center_x", "center_y", "radius", "warp_frequency", "warp_octaves", "warp_seed", "None"]

    def apply_effect_internal(self, image: np.ndarray, warp_type: str, warp_strength: float, 
                              center_x: float, center_y: float, radius: float, 
                              warp_frequency: float = 5.0, warp_octaves: int = 3, 
                              warp_seed: int = 0, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        center = (int(w * center_x), int(h * center_y))
        
        # Create meshgrid
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Calculate distance from center
        dx = x - center[0]
        dy = y - center[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Create a mask based on the radius
        max_dist = np.sqrt(w**2 + h**2)
        mask = np.clip(1 - dist / (radius * max_dist), 0, 1)
        
        if warp_type == "noise":
            # Generate noise for warping
            np.random.seed(warp_seed)
            noise = np.zeros((h, w, 2))
            for _ in range(warp_octaves):
                freq = warp_frequency * (2 ** _)
                amp = warp_strength / (2 ** _)
                # Calculate phase grid
                phase_x = freq * x / w
                phase_y = freq * y / h
                # Generate random noise and modulate with sine waves
                rand_noise = np.random.rand(h, w, 2)
                noise += amp * rand_noise * np.stack((np.sin(phase_y), np.sin(phase_x)), axis=-1)
            
            x_warped = x + noise[:,:,0] * w * mask
            y_warped = y + noise[:,:,1] * h * mask
            
        elif warp_type == "twist":
            angle = np.arctan2(dy, dx)
            twist = warp_strength * dist * mask
            x_warped = x + np.sin(angle + twist) * dist - dx
            y_warped = y - np.cos(angle + twist) * dist - dy
            
        elif warp_type == "bulge":
            bulge = 1 + warp_strength * mask
            x_warped = center[0] + dx * bulge
            y_warped = center[1] + dy * bulge
        
        else:
            raise ValueError(f"Unknown warp type: {warp_type}")
        
        # Ensure warped coordinates are within image bounds
        x_warped = np.clip(x_warped, 0, w-1)
        y_warped = np.clip(y_warped, 0, h-1)
        
        # Remap image
        warped = cv2.remap(image, x_warped.astype(np.float32), y_warped.astype(np.float32), cv2.INTER_LINEAR)
        
        return warped
    

@apply_tooltips
class FlexImageVignette(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "radius": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
            "feather": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["intensity", "radius", "feather", "center_x", "center_y", "None"]

    def apply_effect_internal(self, image: np.ndarray, intensity: float, radius: float, feather: float, 
                              center_x: float, center_y: float, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center = (int(w * center_x), int(h * center_y))
        
        # Calculate distance from center
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Normalize distance
        max_dist = np.sqrt((max(center[0], w - center[0]))**2 + (max(center[1], h - center[1]))**2)
        normalized_dist = dist / max_dist
        
        # Apply radius
        normalized_dist = normalized_dist / radius
        
        # Create vignette mask
        mask = 1 - np.clip(normalized_dist, 0, 1)
        
        # Apply feathering
        mask = np.clip((mask - (1 - feather)) / feather, 0, 1)
        
        # Apply intensity
        mask = mask * intensity + (1 - intensity)
        
        # Reshape mask to match image dimensions
        mask = mask.reshape(h, w, 1)
        
        # Apply vignette
        result = image * mask
        
        return np.clip(result, 0, 1)
    

@apply_tooltips
class FlexImageTransform(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "transform_type": (["translate", "rotate", "scale"],),
            "x_value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
            "y_value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["x_value", "y_value", "None"]

    def apply_effect_internal(self, image: np.ndarray, transform_type: str, x_value: float, y_value: float, **kwargs) -> np.ndarray:
        return transform_image(image, transform_type, x_value, y_value)

@apply_tooltips
class FlexImageHueShift(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "hue_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
        })
        base_inputs["optional"].update({
            "opt_mask": ("MASK",),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["hue_shift", "None"]

    def apply_effect_internal(self, image: np.ndarray, hue_shift: float, opt_mask: np.ndarray = None, **kwargs) -> np.ndarray:
        # Convert RGB to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create a copy of the original HSV image
        result_hsv = hsv_image.copy()

        # Apply hue shift
        result_hsv[:,:,0] = (result_hsv[:,:,0] + hue_shift / 2) % 180

        if opt_mask is not None:
            # Ensure mask has the same shape as the image
            if opt_mask.shape[:2] != image.shape[:2]:
                opt_mask = cv2.resize(opt_mask, (image.shape[1], image.shape[0]))

            # Normalize mask to range [0, 1]
            if opt_mask.max() > 1:
                opt_mask = opt_mask / 255.0

            # Expand mask dimensions to match HSV image
            mask_3d = np.expand_dims(opt_mask, axis=2)

            # Apply the mask
            result_hsv = hsv_image * (1 - mask_3d) + result_hsv * mask_3d

        # Convert back to RGB
        result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)

        return np.clip(result, 0, 1)


import numpy as np
import cv2

@apply_tooltips
class FlexImageDepthWarp(FlexImageBase):

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "warp_strength": ("FLOAT", {"default": 0.1, "min": -10.0, "max": 10.0, "step": 0.01}),
            "depth_map": ("IMAGE",),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["warp_strength", "None"]

    def apply_effect_internal(
        self,
        image: np.ndarray,
        warp_strength: float,
        depth_map: np.ndarray = None,
        frame_index: int = 0,
        **kwargs
    ) -> np.ndarray:
        h, w, _ = image.shape

        if depth_map is not None:
            # Extract the depth map for the current frame
            depth_map_frame = depth_map[frame_index].cpu().numpy()

            # Compute the depth value as the average of the 3 color channels
            depth_map_gray = np.mean(depth_map_frame, axis=-1)

            # Normalize the depth map to [0,1]
            depth_map_normalized = depth_map_gray / np.max(depth_map_gray)

            # Compute displacements based on depth
            # warp_strength controls the maximum displacement in pixels
            dx = warp_strength * (depth_map_normalized - 0.5) * w
            dy = warp_strength * (depth_map_normalized - 0.5) * h

            # Generate coordinate grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            # Apply displacements
            x_displaced = x + dx
            y_displaced = y + dy

            # Ensure coordinates are within image bounds
            x_displaced = np.clip(x_displaced, 0, w - 1)
            y_displaced = np.clip(y_displaced, 0, h - 1)

            # Warp the image using remapping
            map_x = x_displaced.astype(np.float32)
            map_y = y_displaced.astype(np.float32)

            warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

            return np.clip(warped_image, 0, 1)
        else:
            print("Warning: No depth map provided.")
            return image


@apply_tooltips
class FlexImageHorizontalToVertical(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "blur_amount": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            "background_type": (["blur", "border", "mirror", "gradient", "pixelate", "waves"], {"default": "blur"}),
            "border_color": (["black", "white"], {"default": "black"}),
            "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),  # Updated max to 2.0
            "effect_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["blur_amount", "scale_factor", "effect_strength", "None"]

    def apply_effect_internal(self, image: np.ndarray, blur_amount: float, background_type: str, 
                              border_color: str, scale_factor: float, effect_strength: float, **kwargs) -> np.ndarray:
        # Ensure parameters are within valid ranges after modulation
        blur_amount = max(0.1, blur_amount)
        scale_factor = np.clip(scale_factor, 0.1, 2.0)  # Updated max to 2.0
        effect_strength = np.clip(effect_strength, 0.0, 2.0)
        
        h, w = image.shape[:2]
        
        # Only process if image is horizontal
        if w <= h:
            return image
                
        # Calculate new dimensions maintaining aspect ratio
        new_h = h
        new_w = int(h * 9/16)  # Standard vertical aspect ratio (16:9 inverted)
        
        # Calculate scaling for the original image
        scale = scale_factor * min(new_w / w, new_h / h)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        # Create base canvas
        if border_color == "white":
            result = np.ones((new_h, new_w, 3), dtype=np.float32)
        else:  # black
            result = np.zeros((new_h, new_w, 3), dtype=np.float32)
                
        # Apply background effect
        if background_type == "blur":
            background = cv2.resize(image, (new_w, new_h))
            try:
                background = cv2.GaussianBlur(background, (0, 0), blur_amount)
            except cv2.error:
                print(f"Warning: Blur failed with amount {blur_amount}, using minimum blur")
                background = cv2.GaussianBlur(background, (0, 0), 0.1)
            result = background
                
        elif background_type == "mirror":
            # Create mirrored background
            background = cv2.resize(image, (new_w, new_h))
            flipped = cv2.flip(background, 1)
            alpha = np.linspace(0, 1, new_w)
            alpha = np.tile(alpha, (new_h, 1))
            result = background * alpha[:,:,np.newaxis] + flipped * (1 - alpha[:,:,np.newaxis])
                
        elif background_type == "gradient":
            # Create gradient background using original image colors
            resized = cv2.resize(image, (new_w, new_h))
            avg_color = np.mean(resized, axis=(0,1))
            gradient = np.linspace(0, 1, new_h)[:,np.newaxis]
            gradient = np.tile(gradient, (1, new_w))
            for c in range(3):
                result[:,:,c] = gradient * avg_color[c] * effect_strength
                    
        elif background_type == "pixelate":
            # Create heavily pixelated background
            pixel_size = max(1, int(20 * effect_strength))
            small = cv2.resize(image, (new_w // pixel_size, new_h // pixel_size))
            result = cv2.resize(small, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
        elif background_type == "waves":
            # Create wavy background
            background = cv2.resize(image, (new_w, new_h))
            y, x = np.mgrid[0:new_h, 0:new_w]
            frequency = 0.05 * effect_strength
            waves = np.sin(x * frequency) * 10 * effect_strength
            for i in range(new_h):
                shift = int(waves[i][0])
                result[i] = np.roll(background[i], shift, axis=0)
        
        # Scale original image
        scaled_image = cv2.resize(image, (scaled_w, scaled_h))
        
        # Calculate position to place scaled image
        y_offset = (new_h - scaled_h) // 2
        x_offset = (new_w - scaled_w) // 2

        # Compute regions for placement, handling possible cropping
        y_start_img = max(0, -y_offset)
        y_end_img = scaled_h - max(0, (y_offset + scaled_h) - new_h)
        x_start_img = max(0, -x_offset)
        x_end_img = scaled_w - max(0, (x_offset + scaled_w) - new_w)

        y_start_res = max(0, y_offset)
        y_end_res = min(new_h, y_offset + scaled_h)
        x_start_res = max(0, x_offset)
        x_end_res = min(new_w, x_offset + scaled_w)

        # Place the valid region of the scaled image onto the result
        result[y_start_res:y_end_res, x_start_res:x_end_res] = scaled_image[y_start_img:y_end_img, x_start_img:x_end_img]
        
        return np.clip(result, 0, 1)

     