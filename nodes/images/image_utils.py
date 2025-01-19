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

def apply_sharpen(image: np.ndarray, intensity: float, kernel_size: int, sigma: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = cv2.addWeighted(image, 1 + intensity, blurred, -intensity, 0)
    return np.clip(sharpened, 0, 1)

def apply_edge_detect(image: np.ndarray, intensity: float, kernel_size: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(np.uint8(gray * 255), 100, 200)
    edges = edges.astype(float) / 255.0
    return np.clip(image * (1 - intensity) + np.dstack([edges] * 3) * intensity, 0, 1)

def apply_emboss(image: np.ndarray, intensity: float, kernel_size: int) -> np.ndarray:
    kernel = np.array([[-1,-1,0], [-1,0,1], [0,1,1]])
    if kernel_size > 3:
        kernel = cv2.resize(kernel, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    embossed = cv2.filter2D(gray, -1, kernel * intensity)
    embossed = (embossed - np.min(embossed)) / (np.max(embossed) - np.min(embossed))
    return np.clip(image * (1 - intensity) + np.dstack([embossed] * 3) * intensity, 0, 1)

def apply_posterize(image: np.ndarray, intensity: float, levels: int, dither: bool = False) -> np.ndarray:
    effective_levels = int(np.clip(levels * (1 - intensity) + 2 * intensity, 2, levels))
    quantized = np.round(image * (effective_levels - 1)) / (effective_levels - 1)
    if dither:
        error = image - quantized
        quantized += np.random.uniform(-0.5/effective_levels, 0.5/effective_levels, image.shape)
    return np.clip(quantized, 0, 1)

def apply_brightness(image: np.ndarray, intensity: float, midpoint: float = 0.5, preserve_luminosity: bool = False) -> np.ndarray:
    adjusted = image + intensity
    
    if preserve_luminosity:
        original_luminosity = np.mean(image)
        adjusted_luminosity = np.mean(adjusted)
        adjusted *= original_luminosity / adjusted_luminosity
    
    return np.clip(adjusted, 0, 1)

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




def warp_affine(image: np.ndarray, M: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REPLICATE)

def translate_image(image: np.ndarray, x_value: float, y_value: float) -> np.ndarray:
    M = np.float32([[1, 0, x_value],
                    [0, 1, y_value]])
    return warp_affine(image, M)

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    
    # Find the bounding box of non-zero pixels
    rows = np.any(image, axis=(1, 2))
    cols = np.any(image, axis=(0, 2))
    ymin, ymax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, height-1)
    xmin, xmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, width-1)
    
    center_y = int((ymin + ymax) / 2)
    center_x = int((xmin + xmax) / 2)
    
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    rotated = np.zeros_like(image)
    
    image_rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rotated = np.maximum(rotated, image_rotated)
    
    return rotated

def scale_image(image: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    height, width = image.shape[:2]
    center_x, center_y = width / 2, height / 2
    
    M = np.float32([
        [scale_x, 0, center_x * (1 - scale_x)],
        [0, scale_y, center_y * (1 - scale_y)]
    ])
    
    return warp_affine(image, M)

def transform_image(image: np.ndarray, transform_type: str, x_value: float, y_value: float) -> np.ndarray:
    if transform_type == "translate":
        return translate_image(image, x_value, y_value)
    elif transform_type == "rotate":
        return rotate_image(image, x_value)
    elif transform_type == "scale":
        return scale_image(image, 1 + x_value, 1 + y_value)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")