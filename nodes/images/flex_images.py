import cv2
import torch
import numpy as np
from .image_base import FlexImageBase
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from .image_utils import transform_image

class FlexImageEdgeDetect(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "low_threshold": ("FLOAT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("FLOAT", {"default": 200, "min": 0, "max": 255, "step": 1}),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["low_threshold", "high_threshold", "None"]

    def apply_effect_internal(self, image: np.ndarray, low_threshold: float, high_threshold: float, **kwargs) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        edges = cv2.Canny(np.uint8(gray * 255), low_threshold, high_threshold)
        
        # Convert back to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return edges_rgb.astype(float) / 255.0

class FlexImagePosterize(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "max_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
                "dither_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "channel_separation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 2.2, "step": 0.1}),
            }
        }

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
    
class FlexImageKaleidoscope(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "segments": ("INT", {"default": 8, "min": 2, "max": 32, "step": 1}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "precession": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

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
    
class FlexImageColorGrade(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "lut_file": ("STRING", {"default": ""}),
            },
        }

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

class FlexImageChromaticAberration(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "shift_amount": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001}),
                "angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
            }
        }

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

class FlexImagePixelate(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "pixel_size": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
            }
        }

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
    
class FlexImageBloom(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_amount": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

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

class FlexImageTiltShift(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "blur_amount": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "focus_position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_height": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "focus_shape": (["rectangle", "ellipse"], {"default": "rectangle"}),
            }
        }

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
    
 
class FlexImageParallax(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "shift_x": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shift_y": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shift_z": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"depth_map": ("IMAGE",),
            },
        }

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
    
class FlexImageContrast(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "preserve_luminosity": ("BOOLEAN", {"default": True}),
            }
        }

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

class FlexImageWarp(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "warp_type": (["noise", "twist", "bulge"], {"default": "noise"}),
                "warp_strength": ("FLOAT", {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "warp_frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "warp_octaves": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "warp_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

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
                noise += amp * np.random.rand(h, w, 2) * np.sin(freq * np.stack((y, x)) / np.array([h, w]))
            
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
        warped = cv2.remap(image, x_warped, y_warped, cv2.INTER_LINEAR)
        
        return warped
    

class FlexImageVignette(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.01}),
                "feather": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

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
    

class FlexImageTransform(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "transform_type": (["translate", "rotate", "scale"],),
                "x_value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "y_value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["x_value", "y_value", "None"]

    def apply_effect_internal(self, image: np.ndarray, transform_type: str, x_value: float, y_value: float, **kwargs) -> np.ndarray:
        return transform_image(image, transform_type, x_value, y_value)

