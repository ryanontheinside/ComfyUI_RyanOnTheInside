import cv2
import torch
import numpy as np
from .flex_image_base import FlexImageBase
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F
from .image_utils import transform_image, apply_gaussian_blur_gpu
from ...tooltips import apply_tooltips
from ..node_utilities import string_to_rgb

@apply_tooltips
class FlexImageEdgeDetect(FlexImageBase):
    @classmethod

    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "max_levels": ("INT", {"default": 8, "min": 2, "max": 256, "step": 1}),
            "dither_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "channel_separation": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "gamma": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 2.2, "step": 0.1}),
            "dither_method": (["ordered", "floyd", "none"], {"default": "ordered"}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["max_levels", "dither_strength", "channel_separation", "gamma", "None"]

    def apply_effect_internal(self, image: np.ndarray, max_levels: int, dither_strength: float, 
                              channel_separation: float, gamma: float, dither_method: str, **kwargs) -> np.ndarray:
        # Apply gamma correction
        image_gamma = np.power(image, gamma)
        
        # Initialize result array
        result = np.zeros_like(image_gamma)
        
        # Create ordered dither pattern if needed
        if dither_method == "ordered":
            bayer_pattern = np.array([[0, 8, 2, 10],
                                    [12, 4, 14, 6],
                                    [3, 11, 1, 9],
                                    [15, 7, 13, 5]], dtype=np.float32) / 16.0
            h, w = image.shape[:2]
            # Tile the pattern to match image size
            pattern_h = (h + 3) // 4
            pattern_w = (w + 3) // 4
            dither_pattern = np.tile(bayer_pattern, (pattern_h, pattern_w))[:h, :w]
            dither_pattern = dither_pattern[..., np.newaxis] * dither_strength
        
        for c in range(3):  # RGB channels
            # Calculate levels for each channel with separation
            channel_levels = int(np.clip(2 + (max_levels - 2) * (1 + channel_separation * (c - 1)), 2, max_levels))
            
            # Get current channel
            channel = image_gamma[..., c]
            
            if dither_method == "ordered":
                # Add ordered dither pattern
                channel = np.clip(channel + dither_pattern[..., 0] - 0.5 * dither_strength, 0, 1)
            
            # Quantize
            scale = (channel_levels - 1)
            quantized = np.round(channel * scale) / scale
            
            if dither_method == "floyd":
                # Efficient Floyd-Steinberg dithering using convolution
                error = (channel - quantized) * dither_strength
                
                # Prepare error diffusion kernel
                kernel = np.array([[0, 0, 0],
                                [0, 0, 7/16],
                                [3/16, 5/16, 1/16]])
                
                # Apply error diffusion
                from scipy.signal import convolve2d
                error_diffused = convolve2d(error, kernel, mode='same')
                quantized = np.clip(quantized + error_diffused, 0, 1)
            
            result[..., c] = quantized
        
        # Apply inverse gamma correction
        return np.power(result, 1/gamma)
    
@apply_tooltips
class FlexImageKaleidoscope(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "glitch_type": (["digital", "analog", "compression", "wave", "corrupt"], {"default": "digital"}),
            "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "block_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
            "wave_amplitude": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "wave_frequency": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            "corruption_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "time_seed": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["intensity", "block_size", "wave_amplitude", "wave_frequency", "corruption_amount", "time_seed", "None"]

    def apply_effect_internal(self, image: np.ndarray, glitch_type: str, intensity: float, 
                            block_size: int, wave_amplitude: float, wave_frequency: float,
                            corruption_amount: float, time_seed: int, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        
        # Add smart padding - using reflection for most natural look
        pad_size = max(int(min(h, w) * 0.1), 32)  # At least 32 pixels or 10% of size
        padded = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, 
                                  cv2.BORDER_REFLECT_101)
        
        ph, pw = padded.shape[:2]
        result = padded.copy()
        
        # Set random seed for reproducibility
        np.random.seed(time_seed)
        
        # Apply effects as before, but now working with padded image
        if glitch_type == "digital":
            for c in range(3):
                shift_x = int(pw * intensity * np.random.uniform(-0.1, 0.1))
                shift_y = int(ph * intensity * np.random.uniform(-0.1, 0.1))
                result[..., c] = np.roll(np.roll(padded[..., c], shift_x, axis=1), shift_y, axis=0)
            
            num_blocks = int(intensity * 10)
            for _ in range(num_blocks):
                x = np.random.randint(0, pw - block_size)
                y = np.random.randint(0, ph - block_size)
                block = result[y:y+block_size, x:x+block_size].copy()
                
                shift_x = int(block_size * np.random.uniform(-1, 1))
                shift_y = int(block_size * np.random.uniform(-1, 1))
                
                new_x = np.clip(x + shift_x, 0, pw - block_size)
                new_y = np.clip(y + shift_y, 0, ph - block_size)
                
                result[new_y:new_y+block_size, new_x:new_x+block_size] = block
                
        elif glitch_type == "compression":
            # Simulate JPEG compression artifacts
            num_blocks_y = ph // block_size
            num_blocks_x = pw // block_size
            
            for by in range(num_blocks_y):
                for bx in range(num_blocks_x):
                    y1, y2 = by * block_size, (by + 1) * block_size
                    x1, x2 = bx * block_size, (bx + 1) * block_size
                    
                    # Random block corruption
                    if np.random.random() < corruption_amount:
                        # Quantization effect
                        block = result[y1:y2, x1:x2]
                        # Simulate DCT quantization
                        quant_level = int(8 * intensity)
                        block = (block * quant_level).astype(int) / quant_level
                        # Add blocking artifacts
                        block += np.random.uniform(-0.1, 0.1, block.shape) * intensity
                        result[y1:y2, x1:x2] = block
                
        elif glitch_type == "wave":
            y_coords, x_coords = np.mgrid[0:ph, 0:pw]
            
            for i in range(3):
                freq = wave_frequency * (i + 1)
                amp = wave_amplitude / (i + 1)
                
                x_offset = amp * pw * np.sin(2 * np.pi * y_coords / ph * freq + time_seed * 0.1)
                y_offset = amp * ph * np.cos(2 * np.pi * x_coords / pw * freq + time_seed * 0.1)
                
                x_map = (x_coords + x_offset * intensity).astype(np.float32)
                y_map = (y_coords + y_offset * intensity).astype(np.float32)
                
                x_map = np.clip(x_map, 0, pw-1)
                y_map = np.clip(y_map, 0, ph-1)
                
                result = cv2.remap(result, x_map, y_map, cv2.INTER_LINEAR)
        
        elif glitch_type == "corrupt":
            # Data corruption simulation
            for _ in range(int(corruption_amount * 20)):
                # Random line corruption
                if np.random.random() < 0.5:
                    y = np.random.randint(0, ph)
                    length = int(pw * np.random.uniform(0.1, 0.5))
                    start = np.random.randint(0, pw - length)
                    
                    # Corrupt line with various effects
                    corrupt_type = np.random.choice(['repeat', 'shift', 'noise'])
                    if corrupt_type == 'repeat':
                        result[y, start:start+length] = result[y, start]
                    elif corrupt_type == 'shift':
                        shift = np.random.randint(-50, 50)
                        result[y, start:start+length] = np.roll(result[y, start:start+length], shift, axis=0)
                    else:  # noise
                        result[y, start:start+length] = np.random.random((length, 3))
                
                # Block corruption
                else:
                    y = np.random.randint(0, ph - block_size)
                    x = np.random.randint(0, pw - block_size)
                    
                    # Different corruption patterns
                    pattern = np.random.choice(['solid', 'noise', 'repeat'])
                    if pattern == 'solid':
                        result[y:y+block_size, x:x+block_size] = np.random.random(3)
                    elif pattern == 'noise':
                        result[y:y+block_size, x:x+block_size] = np.random.random((block_size, block_size, 3))
                    else:  # repeat
                        result[y:y+block_size, x:x+block_size] = result[y, x]
        
        elif glitch_type == "analog":
            # Simulate analog TV distortion
            # Add scan lines
            scan_lines = np.ones((ph, pw, 3))
            scan_lines[::2] *= 0.8
            result *= scan_lines
            
            # Add noise
            noise = np.random.normal(0, 0.1 * intensity, (ph, pw, 3))
            result += noise
            
            # Add vertical hold distortion
            hold_offset = int(ph * 0.1 * intensity * np.sin(time_seed * 0.1))
            result = np.roll(result, hold_offset, axis=0)
            
            # Add ghosting
            ghost = np.roll(result, int(pw * 0.05 * intensity), axis=1) * 0.3
            result = result * 0.7 + ghost
        
        # Extract the center portion (removing padding)
        result = result[pad_size:pad_size+h, pad_size:pad_size+w]
        
        return np.clip(result, 0, 1)

@apply_tooltips
class FlexImageChromaticAberration(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "shift_amount": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.5, "step": 0.001}),
            "angle": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 720.0, "step": 1.0}),
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            "blur_amount": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "num_passes": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            "color_bleeding": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "falloff": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 3.0, "step": 0.1}),
        })
        base_inputs["optional"].update({
            "opt_normal_map": ("IMAGE",),
            "opt_mask": ("MASK",),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["intensity", "threshold", "blur_amount", "num_passes", "color_bleeding", "falloff", "None"]

    def __init__(self):
        super().__init__()
        self.kernel_cache = {}
        self.weights_cache = {}
        self.device = None

    def _get_device(self):
        """Get or initialize device (GPU if available, CPU if not)"""
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self.device

    def _get_pass_weights(self, num_passes):
        """Get cached pass weights or create new ones"""
        key = num_passes
        if key not in self.weights_cache:
            device = self._get_device()
            weights = torch.tensor([1.0 / (2 ** i) for i in range(num_passes)], device=device)
            self.weights_cache[key] = weights / weights.sum()
        return self.weights_cache[key]

    def _prepare_mask(self, opt_mask, bright_mask_shape, frame_index):
        """Prepare and transform mask for bloom effect"""
        device = self._get_device()
        
        # Skip if no mask provided
        if opt_mask is None:
            return None
            
        # Convert mask to tensor if needed
        if not torch.is_tensor(opt_mask):
            mask_tensor = torch.from_numpy(opt_mask).to(device)
        else:
            mask_tensor = opt_mask.to(device)
        
        # Extract the correct frame from batched masks
        if len(mask_tensor.shape) > 2:
            mask_tensor = mask_tensor[frame_index]
        
        # Ensure mask is 2D
        if len(mask_tensor.shape) > 2:
            mask_tensor = mask_tensor.squeeze()
        
        # Only resize if necessary
        if mask_tensor.shape != bright_mask_shape:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=bright_mask_shape,
                mode='bilinear'
            ).squeeze()
            
        return mask_tensor

    def _prepare_normal_map(self, opt_normal_map, frame_index):
        """Prepare normal map for directional bloom"""
        device = self._get_device()
        
        # Skip if no normal map provided
        if opt_normal_map is None:
            return None
            
        # Convert to tensor if needed
        if not torch.is_tensor(opt_normal_map):
            normal_tensor = torch.from_numpy(opt_normal_map).to(device)
        else:
            normal_tensor = opt_normal_map.to(device)
        
        # Extract the correct frame
        if len(normal_tensor.shape) > 3:
            normal_tensor = normal_tensor[frame_index]
        
        # Convert normal map to [-1,1] range
        normals = normal_tensor * 2.0 - 1.0
        
        # Calculate surface alignment
        view_vector = torch.tensor([0, 0, 1], device=device)
        surface_alignment = torch.sum(normals * view_vector, dim=2)
        surface_alignment = (surface_alignment + 1) * 0.5
        
        return surface_alignment

    def apply_effect_internal(self, image: np.ndarray, threshold: float, blur_amount: float, 
                              intensity: float, num_passes: int, color_bleeding: float,
                              falloff: float, opt_normal_map: np.ndarray = None, 
                              opt_mask: np.ndarray = None, **kwargs) -> np.ndarray:
        # Skip processing if intensity is 0
        if intensity <= 0.001:
            return image
            
        # Skip processing if blur amount is 0
        if blur_amount <= 0.001:
            return image
        
        # Get device and convert input to tensor
        device = self._get_device()
        frame_index = kwargs.get('frame_index', 0)
        
        # Convert image to tensor efficiently
        image_tensor = torch.from_numpy(image).to(device)
        
        # Extract bright areas with smooth threshold - vectorized operation
        brightness = torch.max(image_tensor, dim=2)[0]
        # Only process if brightness exceeds threshold
        if torch.max(brightness) <= threshold:
            return image
            
        # Calculate bright mask with threshold
        bright_mask = torch.clamp((brightness - threshold) / max(1e-6, 1 - threshold), 0, 1)
        bright_mask = torch.pow(bright_mask, falloff)
        
        # Process mask if provided
        mask_tensor = self._prepare_mask(opt_mask, bright_mask.shape, frame_index)
        if mask_tensor is not None:
            bright_mask = bright_mask * mask_tensor
            
        # Skip if bright mask is empty after masking
        if torch.max(bright_mask) <= 0.001:
            return image
        
        # Calculate color bleeding contribution
        if color_bleeding > 0:
            mean_color = torch.mean(image_tensor, dim=2, keepdim=True)
            color_contribution = image_tensor * (1 - color_bleeding) + mean_color * color_bleeding
        else:
            color_contribution = image_tensor
        
        # Initialize bloom accumulator
        bloom_accumulator = torch.zeros_like(image_tensor)
        
        # Get pass weights
        pass_weights = self._get_pass_weights(num_passes)
        
        # Process normal map if provided
        surface_alignment = self._prepare_normal_map(opt_normal_map, frame_index)
        
        # Multi-pass gaussian blur
        for i in range(num_passes):
            # Calculate adaptive kernel size for this pass
            kernel_size = int(blur_amount * (1 + i)) | 1  # Ensure odd
            kernel_size = max(3, min(kernel_size, min(image.shape[:2])))
            sigma = kernel_size / 6.0
            
            # Skip if the weight contribution would be negligible
            if pass_weights[i] < 0.01:
                continue
                
            # Apply bright mask with color contribution
            bright_mask_expanded = bright_mask.unsqueeze(-1)
            pass_contribution = color_contribution * bright_mask_expanded
            
            # Apply gaussian blur efficiently
            pass_contribution = apply_gaussian_blur_gpu(
                pass_contribution.permute(2, 0, 1),
                kernel_size,
                sigma
            ).permute(1, 2, 0)
            
            # Modulate by surface alignment if normal map is provided
            if surface_alignment is not None:
                pass_contribution = pass_contribution * (1 - surface_alignment.unsqueeze(-1))
            
            # Add weighted contribution to accumulator
            bloom_accumulator.add_(pass_contribution * pass_weights[i])
        
        # Combine with original image using intensity
        result = image_tensor + bloom_accumulator * intensity
        
        # Convert back to numpy and ensure range
        result = result.cpu().numpy()
        return np.clip(result, 0, 1)
    
@apply_tooltips
class FlexImageTiltShift(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "blur_amount": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.1}),
            "focus_position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "focus_position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "focus_width": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "focus_height": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "focus_shape": (["rectangle", "ellipse", "gradient"], {"default": "gradient"}),
            "bokeh_shape": (["circular", "hexagonal", "star"], {"default": "circular"}),
            "bokeh_size": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
            "bokeh_brightness": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "step": 0.1}),
            "chromatic_aberration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["blur_amount", "focus_position_x", "focus_position_y", "focus_width", "focus_height", 
                "bokeh_size", "bokeh_brightness", "chromatic_aberration", "None"]

    def _create_bokeh_kernel(self, size, shape, brightness):
        # Create bokeh kernel based on shape
        kernel_size = int(size * 20) | 1  # Ensure odd size
        center = kernel_size // 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        if shape == "circular":
            y, x = np.ogrid[-center:center+1, -center:center+1]
            mask = x*x + y*y <= center*center
            kernel[mask] = 1.0
            
        elif shape == "hexagonal":
            for y in range(kernel_size):
                for x in range(kernel_size):
                    # Hexagonal distance calculation
                    dx = abs(x - center)
                    dy = abs(y - center)
                    if dx * 0.866025 + dy * 0.5 <= center:
                        kernel[y, x] = 1.0
                        
        elif shape == "star":
            for y in range(kernel_size):
                for x in range(kernel_size):
                    dx = x - center
                    dy = y - center
                    angle = np.arctan2(dy, dx)
                    dist = np.sqrt(dx*dx + dy*dy)
                    # Create 6-point star shape
                    star_factor = np.abs(np.sin(3 * angle))
                    if dist <= center * (0.8 + 0.2 * star_factor):
                        kernel[y, x] = 1.0
        
        # Normalize and apply brightness
        kernel = kernel / np.sum(kernel) * brightness
        return kernel

    def apply_effect_internal(self, image: np.ndarray, blur_amount: float, focus_position_x: float, 
                              focus_position_y: float, focus_width: float, focus_height: float, 
                              focus_shape: str, bokeh_shape: str, bokeh_size: float,
                              bokeh_brightness: float, chromatic_aberration: float, **kwargs) -> np.ndarray:
        h, w = image.shape[:2]
        center_x, center_y = int(w * focus_position_x), int(h * focus_position_y)
        width, height = int(w * focus_width), int(h * focus_height)

        # Create focus mask
        mask = np.zeros((h, w), dtype=np.float32)
        if focus_shape == "rectangle":
            cv2.rectangle(mask, 
                          (center_x - width//2, center_y - height//2),
                          (center_x + width//2, center_y + height//2),
                          1, -1)
        elif focus_shape == "ellipse":
            cv2.ellipse(mask, 
                        (center_x, center_y),
                        (width//2, height//2),
                        0, 0, 360, 1, -1)
        else:  # gradient
            y, x = np.ogrid[0:h, 0:w]
            # Create smooth gradient based on distance from focus center
            dx = (x - center_x) / (width/2)
            dy = (y - center_y) / (height/2)
            dist = np.sqrt(dx*dx + dy*dy)
            mask = np.clip(1 - dist, 0, 1)

        # Apply gaussian blur to mask for smooth transition
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=min(width, height) / 6)

        # Create bokeh kernel
        bokeh_kernel = self._create_bokeh_kernel(bokeh_size, bokeh_shape, bokeh_brightness)
        
        # Process each channel separately for chromatic aberration
        result = np.zeros_like(image)
        for c in range(3):
            # Apply channel-specific blur offset for chromatic aberration
            offset = (c - 1) * chromatic_aberration * blur_amount
            channel_blur = cv2.filter2D(image[..., c], -1, bokeh_kernel * (blur_amount + offset))
            result[..., c] = image[..., c] * mask + channel_blur * (1 - mask)

        return np.clip(result, 0, 1)
    
@apply_tooltips
class FlexImageParallax(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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

    def __init__(self):
        super().__init__()
        self._cached_coords = {}  # Cache for coordinate grids

    def _get_coordinate_grid(self, h, w):
        """Get cached coordinate grid or create a new one"""
        key = (h, w)
        if key not in self._cached_coords:
            y, x = np.mgrid[0:h, 0:w].astype(np.float32)
            self._cached_coords[key] = (x, y)
        return self._cached_coords[key]

    def apply_effect_internal(
        self,
        image: np.ndarray,
        shift_x: float,
        shift_y: float,
        shift_z: float,
        depth_map: np.ndarray = None,
        frame_index: int = 0,
        **kwargs
    ) -> np.ndarray:
        h, w, _ = image.shape

        # Get cached coordinate grid
        x, y = self._get_coordinate_grid(h, w)
        
        # Define center once
        cx, cy = w / 2, h / 2

        if depth_map is not None:
            # Get the depth map for this frame
            depth_frame = depth_map[frame_index].cpu().numpy()
            
            # Convert to grayscale by averaging channels
            depth_gray = np.mean(depth_frame, axis=-1)
            
            # Normalize safely
            max_depth = np.max(depth_gray)
            if max_depth > 0:
                depth_normalized = depth_gray / max_depth
            else:
                depth_normalized = depth_gray

            # Calculate displacements based on depth
            dx = w * shift_x * depth_normalized
            dy = h * shift_y * depth_normalized
            scale_factor = 1 + shift_z * depth_normalized
        else:
            # Uniform displacement when no depth map
            dx = np.full((h, w), shift_x * w, dtype=np.float32)
            dy = np.full((h, w), shift_y * h, dtype=np.float32)
            scale_factor = np.full((h, w), 1 + shift_z, dtype=np.float32)

        # Apply shifts
        x_shifted = x + dx
        y_shifted = y + dy

        # Vectorized scaling around center
        x_scaled = cx + (x_shifted - cx) * scale_factor
        y_scaled = cy + (y_shifted - cy) * scale_factor

        # Create maps for cv2.remap - more efficient than simple indexing
        map_x = np.clip(x_scaled, 0, w - 1).astype(np.float32)
        map_y = np.clip(y_scaled, 0, h - 1).astype(np.float32)

        # Use cv2.remap for better interpolation
        result = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        return result
    
@apply_tooltips
class FlexImageContrast(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "transform_type": (["translate", "rotate", "scale"],),
            "x_value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
            "y_value": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
            "edge_mode": (["extend", "wrap", "reflect", "none"],),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["x_value", "y_value", "None"]

    def apply_effect_internal(self, image: np.ndarray, transform_type: str, x_value: float, y_value: float, edge_mode: str, **kwargs) -> np.ndarray:
        return transform_image(image, transform_type, x_value, y_value, edge_mode)

@apply_tooltips
class FlexImageHueShift(FlexImageBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "hue_shift": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
        })
        base_inputs["optional"].update({
            "opt_mask": ("MASK",),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["hue_shift", "None"]

    def apply_effect_internal(self, image: np.ndarray, hue_shift: float, opt_mask: np.ndarray = None, **kwargs) -> np.ndarray:
        # Convert to float32 for better precision
        image = image.astype(np.float32)

        # Convert RGB to LCH color space for better hue manipulation
        # First convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Convert LAB to LCH (cylindrical color space)
        L = lab[:,:,0]
        a = lab[:,:,1]
        b = lab[:,:,2]
        
        # Calculate C (chroma) and H (hue) from a,b
        C = np.sqrt(np.square(a) + np.square(b))
        H = np.arctan2(b, a)
        
        # Apply hue shift in radians (convert from degrees)
        H_shifted = H + (hue_shift * np.pi / 180.0)
        
        # Convert back to LAB
        a_new = C * np.cos(H_shifted)
        b_new = C * np.sin(H_shifted)
        
        # Reconstruct LAB image
        lab_shifted = np.stack([L, a_new, b_new], axis=2)
        
        # Convert back to RGB
        result = cv2.cvtColor(lab_shifted, cv2.COLOR_LAB2RGB)

        if opt_mask is not None:
            # Convert mask to numpy if it's a tensor
            if torch.is_tensor(opt_mask):
                opt_mask = opt_mask.cpu().numpy()

            # Select the correct frame from the mask batch
            frame_index = kwargs.get('frame_index', 0)
            if len(opt_mask.shape) > 2:
                opt_mask = opt_mask[frame_index]

            # Ensure mask is 2D
            if len(opt_mask.shape) > 2:
                opt_mask = opt_mask.squeeze()

            # Get target dimensions and resize if needed
            target_height, target_width = image.shape[:2]
            if opt_mask.shape != (target_height, target_width):
                opt_mask = cv2.resize(opt_mask.astype(np.float32), (target_width, target_height))

            # Normalize mask to range [0, 1]
            if opt_mask.max() > 1:
                opt_mask = opt_mask / 255.0

            # Expand mask dimensions to match image
            mask_3d = np.expand_dims(opt_mask, axis=2)

            # Apply the mask by blending original and shifted images
            result = image * (1 - mask_3d) + result * mask_3d

        return np.clip(result, 0, 1)



import numpy as np
import cv2

@apply_tooltips
class FlexImageDepthWarp(FlexImageBase):

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
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


     