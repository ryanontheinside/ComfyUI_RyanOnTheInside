import numpy as np
import cv2
import torch

def apply_blend_mode(base_image: np.ndarray, blend_image: np.ndarray, mode: str, opacity: float) -> np.ndarray:
    """
    Apply various blend modes to combine two images.
    
    :param base_image: The bottom layer image
    :param blend_image: The top layer image to blend
    :param mode: The blending mode to use
    :param opacity: The opacity of the blend (0.0 to 1.0)
    :return: The blended image
    """
    if mode == "normal":
        blended = base_image * (1 - opacity) + blend_image * opacity
    elif mode == "multiply":
        blended = base_image * blend_image
    elif mode == "screen":
        blended = 1 - (1 - base_image) * (1 - blend_image)
    elif mode == "overlay":
        mask = base_image >= 0.5
        blended = np.where(mask, 1 - 2 * (1 - base_image) * (1 - blend_image), 2 * base_image * blend_image)
    elif mode == "soft_light":
        mask = blend_image <= 0.5
        blended = np.where(mask, 
                           base_image - (1 - 2 * blend_image) * base_image * (1 - base_image),
                           base_image + (2 * blend_image - 1) * (np.sqrt(base_image) - base_image))
    else:
        raise ValueError(f"Unsupported blend mode: {mode}")
    
    return base_image * (1 - opacity) + blended * opacity

def preserve_original_colors(original_image: np.ndarray, processed_image: np.ndarray) -> np.ndarray:
    """
    Preserve the colors of the original image while keeping the luminance of the processed image.
    
    :param original_image: The original image
    :param processed_image: The processed image whose luminance we want to keep
    :return: An image with original colors and processed luminance
    """
    def rgb_to_hsl(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        max_rgb = np.max(rgb, axis=-1)
        min_rgb = np.min(rgb, axis=-1)
        diff = max_rgb - min_rgb

        l = (max_rgb + min_rgb) / 2

        s = np.zeros_like(l)
        mask = diff != 0
        s[mask] = diff[mask] / (1 - np.abs(2 * l[mask] - 1))

        h = np.zeros_like(l)
        mask_r = (max_rgb == r) & (diff != 0)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
        mask_g = (max_rgb == g) & (diff != 0)
        h[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2
        mask_b = (max_rgb == b) & (diff != 0)
        h[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4
        h /= 6

        return np.stack([h, s, l], axis=-1)

    def hsl_to_rgb(hsl):
        h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
        c = (1 - np.abs(2 * l - 1)) * s
        x = c * (1 - np.abs((h * 6) % 2 - 1))
        m = l - c / 2

        rgb = np.zeros_like(hsl)
        mask = (h < 1/6)
        rgb[mask] = [c[mask], x[mask], 0]
        mask = (1/6 <= h) & (h < 2/6)
        rgb[mask] = [x[mask], c[mask], 0]
        mask = (2/6 <= h) & (h < 3/6)
        rgb[mask] = [0, c[mask], x[mask]]
        mask = (3/6 <= h) & (h < 4/6)
        rgb[mask] = [0, x[mask], c[mask]]
        mask = (4/6 <= h) & (h < 5/6)
        rgb[mask] = [x[mask], 0, c[mask]]
        mask = (5/6 <= h)
        rgb[mask] = [c[mask], 0, x[mask]]

        return rgb + m[..., np.newaxis]

    original_hsl = rgb_to_hsl(original_image[..., :3])
    processed_hsl = rgb_to_hsl(processed_image[..., :3])

    result_hsl = np.copy(original_hsl)
    result_hsl[..., 2] = processed_hsl[..., 2]  # Use luminance from processed image

    result_rgb = hsl_to_rgb(result_hsl)

    if original_image.shape[-1] == 4:  # If there's an alpha channel
        result = np.dstack((result_rgb, original_image[..., 3]))
    else:
        result = result_rgb

    return result



def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an array to the range [0, 1].
    
    :param arr: Input array
    :return: Normalized array
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    return arr

def apply_blur(image: np.ndarray, intensity: float, kernel_size: int, sigma: float = 0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred

def create_gaussian_kernel_gpu(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for GPU-accelerated blur operations.
    
    :param kernel_size: Size of the kernel (must be odd)
    :param sigma: Standard deviation of the Gaussian
    :param device: PyTorch device (GPU or CPU)
    :return: Gaussian kernel tensor ready for convolution
    """
    # Create 1D Gaussian kernel
    x = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size, device=device)
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = gauss / gauss.sum()
    
    # Create 2D kernel by outer product
    kernel = kernel.unsqueeze(0) * kernel.unsqueeze(1)
    
    # Normalize and reshape for conv2d
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel.repeat(3, 1, 1, 1)  # Repeat for RGB channels

def apply_gaussian_blur_gpu(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur using GPU acceleration via PyTorch.
    
    :param x: Input tensor in format (C, H, W) or (N, C, H, W)
    :param kernel_size: Size of the Gaussian kernel (must be odd)
    :param sigma: Standard deviation of the Gaussian
    :return: Blurred tensor in same format as input
    """
    if kernel_size < 3:
        return x
        
    # Ensure input is in the right format (N, C, H, W)
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    
    # Create gaussian kernel
    kernel = create_gaussian_kernel_gpu(kernel_size, sigma, x.device)
    
    # Apply padding to prevent border artifacts
    pad_size = kernel_size // 2
    x_padded = torch.nn.functional.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # Apply convolution for each channel
    groups = x.shape[1]  # Number of channels
    blurred = torch.nn.functional.conv2d(x_padded, kernel, groups=groups, padding=0)
    
    return blurred.squeeze(0) if len(x.shape) == 4 else blurred



def apply_contrast(image: np.ndarray, intensity: float, midpoint: float = 0.5, preserve_luminosity: bool = False) -> np.ndarray:
    factor = 1 + intensity
    adjusted = (image - midpoint) * factor + midpoint
    
    if preserve_luminosity:
        original_luminosity = np.mean(image)
        adjusted_luminosity = np.mean(adjusted)
        adjusted *= original_luminosity / adjusted_luminosity
    
    return np.clip(adjusted, 0, 1)

def apply_saturation(image: np.ndarray, intensity: float, preserve_luminosity: bool = True) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + intensity), 0, 1)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    if preserve_luminosity:
        original_luminosity = np.mean(image)
        adjusted_luminosity = np.mean(adjusted)
        adjusted *= original_luminosity / adjusted_luminosity
    
    return np.clip(adjusted, 0, 1)

def apply_hue_shift(image: np.ndarray, intensity: float, preserve_luminosity: bool = True, hue_method: str = "circular") -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    if hue_method == "linear":
        hsv[..., 0] = np.clip(hsv[..., 0] + intensity, 0, 1)
    elif hue_method == "circular":
        hsv[..., 0] = (hsv[..., 0] + intensity) % 1.0
    else:
        raise ValueError(f"Unsupported hue shift method: {hue_method}")
    
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    if preserve_luminosity:
        original_luminosity = np.mean(image)
        adjusted_luminosity = np.mean(adjusted)
        adjusted *= original_luminosity / adjusted_luminosity
    
    return np.clip(adjusted, 0, 1)

def warp_affine(image: np.ndarray, M: np.ndarray, edge_mode: str = "extend") -> np.ndarray:
    height, width = image.shape[:2]
    
    # Map edge modes to cv2 border modes
    border_modes = {
        "extend": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
        "reflect": cv2.BORDER_REFLECT,
        "none": cv2.BORDER_CONSTANT
    }
    
    if edge_mode not in border_modes:
        raise ValueError(f"Unsupported edge mode: {edge_mode}. Supported modes: {list(border_modes.keys())}")
    
    border_mode = border_modes[edge_mode]
    
    if edge_mode == "none":
        return cv2.warpAffine(image, M, (width, height), borderMode=border_mode, borderValue=0)
    else:
        return cv2.warpAffine(image, M, (width, height), borderMode=border_mode)

def translate_image(image: np.ndarray, x_value: float, y_value: float, edge_mode: str = "extend") -> np.ndarray:
    M = np.float32([[1, 0, x_value],
                    [0, 1, y_value]])
    return warp_affine(image, M, edge_mode)

def rotate_image(image: np.ndarray, angle: float, edge_mode: str = "none") -> np.ndarray:
    height, width = image.shape[:2]
    center_x, center_y = width / 2, height / 2
    
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    return warp_affine(image, M, edge_mode)

def scale_image(image: np.ndarray, scale_x: float, scale_y: float, edge_mode: str = "extend") -> np.ndarray:
    height, width = image.shape[:2]
    center_x, center_y = width / 2, height / 2
    
    M = np.float32([
        [scale_x, 0, center_x * (1 - scale_x)],
        [0, scale_y, center_y * (1 - scale_y)]
    ])
    
    return warp_affine(image, M, edge_mode)

def transform_image(image: np.ndarray, transform_type: str, x_value: float, y_value: float, edge_mode: str = "extend") -> np.ndarray:
    """
    Apply various transformations to an image with configurable edge handling.
    
    :param image: Input image as numpy array
    :param transform_type: Type of transform ("translate", "rotate", "scale")
    :param x_value: X parameter for the transform
    :param y_value: Y parameter for the transform (unused for rotation)
    :param edge_mode: How to handle edges - "extend" (replicate), "wrap" (tile), "reflect" (mirror), or "none" (black)
    :return: Transformed image
    """
    if transform_type == "translate":
        return translate_image(image, x_value, y_value, edge_mode)
    elif transform_type == "rotate":
        return rotate_image(image, x_value, edge_mode)
    elif transform_type == "scale":
        return scale_image(image, 1 + x_value, 1 + y_value, edge_mode)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

def create_wave_distortion_map(height: int, width: int, frequency: float, amplitude: float, device: torch.device) -> torch.Tensor:
    """
    Create a wave distortion displacement map for image warping.
    
    :param height: Image height
    :param width: Image width
    :param frequency: Wave frequency
    :param amplitude: Wave amplitude
    :param device: PyTorch device (GPU/CPU)
    :return: Displacement map tensor
    """
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    displacement = amplitude * torch.sin(2 * np.pi * frequency * y.float())
    return displacement

def extract_and_move_blocks(image: torch.Tensor, block_size: int, shift_range: float, device: torch.device) -> torch.Tensor:
    """
    Extract and randomly move blocks in the image.
    
    :param image: Input tensor (H,W,C)
    :param block_size: Size of blocks to move
    :param shift_range: Maximum shift distance as fraction of image size
    :param device: PyTorch device (GPU/CPU)
    :return: Image with moved blocks
    """
    h, w = image.shape[:2]
    result = image.clone()
    
    # Calculate maximum shift in pixels
    max_shift = int(min(h, w) * shift_range)
    
    # Create block positions
    y_blocks = torch.arange(0, h - block_size + 1, block_size, device=device)
    x_blocks = torch.arange(0, w - block_size + 1, block_size, device=device)
    
    # Random shifts for each block
    shifts_y = torch.randint(-max_shift, max_shift + 1, (len(y_blocks), len(x_blocks)), device=device)
    shifts_x = torch.randint(-max_shift, max_shift + 1, (len(y_blocks), len(x_blocks)), device=device)
    
    # Apply shifts using tensor operations
    for i, y in enumerate(y_blocks):
        for j, x in enumerate(x_blocks):
            # Source block coordinates
            y1, y2 = y, y + block_size
            x1, x2 = x, x + block_size
            
            # Target coordinates with shift
            new_y = torch.clamp(y + shifts_y[i,j], 0, h - block_size)
            new_x = torch.clamp(x + shifts_x[i,j], 0, w - block_size)
            
            # Move block
            result[new_y:new_y+block_size, new_x:new_x+block_size] = image[y1:y2, x1:x2]
    
    return result

def apply_compression_artifacts(image: torch.Tensor, block_size: int, quality: float, device: torch.device) -> torch.Tensor:
    """
    Simulate compression artifacts using block-wise operations.
    
    :param image: Input tensor (H,W,C)
    :param block_size: Size of compression blocks
    :param quality: Quality factor (0-1)
    :param device: PyTorch device (GPU/CPU)
    :return: Image with compression artifacts
    """
    h, w = image.shape[:2]
    result = image.clone()
    
    # Quantization levels based on quality
    levels = int(2 + (1 - quality) * 14)  # 2-16 levels
    
    # Process blocks
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y+block_size, x:x+block_size]
            
            # Quantize block values
            block_q = torch.round(block * (levels - 1)) / (levels - 1)
            
            # Add block edges
            edge_strength = (1 - quality) * 0.1
            block_q += edge_strength * (torch.rand_like(block_q) - 0.5)
            
            result[y:y+block_size, x:x+block_size] = block_q
    
    return torch.clamp(result, 0, 1)

def apply_line_corruption(image: torch.Tensor, corruption_probability: float, device: torch.device) -> torch.Tensor:
    """
    Apply random line corruption effects.
    
    :param image: Input tensor (H,W,C)
    :param corruption_probability: Probability of line corruption
    :param device: PyTorch device (GPU/CPU)
    :return: Image with corrupted lines
    """
    h, w = image.shape[:2]
    result = image.clone()
    
    # Generate random line positions
    line_mask = torch.rand(h, device=device) < corruption_probability
    
    # Different corruption types for selected lines
    for y in torch.where(line_mask)[0]:
        corruption_type = torch.randint(0, 3, (1,), device=device)
        
        if corruption_type == 0:  # Shift
            shift = torch.randint(-w//4, w//4 + 1, (1,), device=device)
            result[y] = torch.roll(image[y], shift.item(), dims=0)
        elif corruption_type == 1:  # Noise
            result[y] = torch.rand_like(image[y])
        else:  # Repeat
            result[y] = image[y].roll(1, dims=0)
    
    return result