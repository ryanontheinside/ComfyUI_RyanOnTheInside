"""Tooltips for images-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for images nodes"""

    # FlexImageBase tooltips (inherits from: RyanOnTheInside, FlexBase)
    TooltipManager.register_tooltips("FlexImageBase", {
        "images": "Input image sequence to be processed (IMAGE type)"
        # All other parameters are inherited from FlexBase
    }, inherits_from=['RyanOnTheInside', 'FlexBase'])

    # FlexImageEdgeDetect tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageEdgeDetect", {
        "low_threshold": "Lower bound for the hysteresis thresholding (0 to 255)",
        "high_threshold": "Upper bound for the hysteresis thresholding (0 to 255)"
    }, inherits_from='FlexImageBase')

    # FlexImagePosterize tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImagePosterize", {
        "max_levels": "Maximum number of color levels per channel (2 to 256)",
        "dither_strength": "Intensity of dithering effect (0.0 to 1.0)",
        "channel_separation": "Degree of separation between color channels (0.0 to 1.0)",
        "gamma": "Gamma correction applied before posterization (0.1 to 2.2)"
    }, inherits_from='FlexImageBase')

    # FlexImageKaleidoscope tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageKaleidoscope", {
        "segments": "Number of mirror segments (2 to 32)",
        "center_x": "X-coordinate of the effect center (0.0 to 1.0)",
        "center_y": "Y-coordinate of the effect center (0.0 to 1.0)",
        "zoom": "Zoom factor for the effect (0.1 to 2.0)",
        "rotation": "Rotation angle of the effect (0.0 to 360.0)",
        "precession": "Rate of rotation change over time (-1.0 to 1.0)",
        "speed": "Speed of the effect animation (0.1 to 5.0)"
    }, inherits_from='FlexImageBase')

    # FlexImageColorGrade tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageColorGrade", {
        "intensity": "Strength of the color grading effect (0.0 to 1.0)",
        "mix": "Blend factor between original and graded image (0.0 to 1.0)",
        "lut_file": "Path to the LUT file"
    }, inherits_from='FlexImageBase')

    # FlexImageGlitch tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageGlitch", {
        "shift_amount": "Magnitude of horizontal shift (0.0 to 1.0)",
        "scan_lines": "Number of scan lines to add (0 to 100)",
        "color_shift": "Amount of color channel separation (0.0 to 1.0)"
    }, inherits_from='FlexImageBase')

    # FlexImageChromaticAberration tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageChromaticAberration", {
        "shift_amount": "Magnitude of color channel shift (0.0 to 0.1)",
        "angle": "Angle of the shift effect (0.0 to 360.0)"
    }, inherits_from='FlexImageBase')

    # FlexImagePixelate tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImagePixelate", {
        "pixel_size": "Size of each pixelated block (1 to 100)"
    }, inherits_from='FlexImageBase')

    # FlexImageBloom tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageBloom", {
        "threshold": "Brightness threshold for the bloom effect (0.0 to 1.0)",
        "blur_amount": "Amount of blur applied to the bloom (0.0 to 50.0)",
        "intensity": "Strength of the bloom effect (0.0 to 1.0)"
    }, inherits_from='FlexImageBase')

    # FlexImageTiltShift tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageTiltShift", {
        "blur_amount": "Strength of the blur effect (0.0 to 50.0)",
        "focus_position_x": "X-coordinate of the focus center (0.0 to 1.0)",
        "focus_position_y": "Y-coordinate of the focus center (0.0 to 1.0)",
        "focus_width": "Width of the focus area (0.0 to 1.0)",
        "focus_height": "Height of the focus area (0.0 to 1.0)",
        "focus_shape": "Shape of the focus area ('rectangle' or 'ellipse')"
    }, inherits_from='FlexImageBase')

    # FlexImageParallax tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageParallax", {
        "shift_x": "Horizontal shift factor for the parallax effect (-1.0 to 1.0)",
        "shift_y": "Vertical shift factor for the parallax effect (-1.0 to 1.0)",
        "shift_z": "Z-axis shift factor for the parallax effect (-1.0 to 1.0)",
        "depth_map": "Optional depth map for 3D parallax effect"
    }, inherits_from='FlexImageBase')

    # FlexImageContrast tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageContrast", {
        "contrast": "Controls the amount of contrast adjustment (0.0 to 3.0)",
        "brightness": "Adjusts the overall brightness of the image (-1.0 to 1.0)",
        "preserve_luminosity": "When enabled, maintains the overall luminosity of the image"
    }, inherits_from='FlexImageBase')

    # FlexImageWarp tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageWarp", {
        "warp_type": "Type of warping effect ('noise', 'twist', 'bulge')",
        "warp_strength": "Strength of the warping effect (-1.0 to 1.0)",
        "center_x": "X-coordinate of the warp center (0.0 to 1.0)",
        "center_y": "Y-coordinate of the warp center (0.0 to 1.0)",
        "radius": "Radius of the warp effect (0.0 to 2.0)",
        "warp_frequency": "Optional frequency of the warp effect (0.1 to 20.0)",
        "warp_octaves": "Optional number of noise octaves (1 to 5)",
        "warp_seed": "Optional seed for noise generation"
    }, inherits_from='FlexImageBase')

    # FlexImageVignette tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageVignette", {
        "intensity": "Strength of the vignette effect (0.0 to 1.0)",
        "radius": "Radius of the vignette (0.1 to 2.0)",
        "feather": "Amount of feathering on the vignette edge (0.0 to 1.0)",
        "center_x": "X-coordinate of the vignette center (0.0 to 1.0)",
        "center_y": "Y-coordinate of the vignette center (0.0 to 1.0)"
    }, inherits_from='FlexImageBase')

    # FlexImageTransform tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageTransform", {
        "transform_type": "Type of transformation ('translate', 'rotate', 'scale')",
        "x_value": "X-axis transformation value (-1000.0 to 1000.0)",
        "y_value": "Y-axis transformation value (-1000.0 to 1000.0)"
    }, inherits_from='FlexImageBase')

    # FlexImageHueShift tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageHueShift", {
        "hue_shift": "Amount of hue shift to apply (0.0 to 360.0)",
        "opt_mask": "Optional mask to apply the effect selectively"
    }, inherits_from='FlexImageBase')

    # FlexImageDepthWarp tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageDepthWarp", {
        "warp_strength": "Strength of the warping effect (-10.0 to 10.0)",
        "depth_map": "Depth map for warping the image"
    }, inherits_from='FlexImageBase')

    # FlexImageHorizontalToVertical tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageHorizontalToVertical", {
        "blur_amount": "Amount of blur for background when using blur effect (0.1 to 100.0)",
        "background_type": "Type of background effect ('blur', 'border', 'mirror', 'gradient', 'pixelate', 'waves')",
        "border_color": "Color of the border when using border background type ('black', 'white')",
        "scale_factor": "Scale factor for the main image (0.1 to 2.0)",
        "effect_strength": "Intensity of the background effect (0.0 to 2.0)"
    }, inherits_from='FlexImageBase')

    # ImageUtilityNode tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("ImageUtilityNode", {
        # Base class for image utility nodes
    }, inherits_from='RyanOnTheInside')

    # DyeImage tooltips (inherits from: ImageUtilityNode)
    TooltipManager.register_tooltips("DyeImage", {
        "image": "Input image to be processed (IMAGE type)",
        "source_rgb": "Source RGB color to replace in format 'R,G,B' (default: '255,255,255')",
        "target_rgb": "Target RGB color to replace with in format 'R,G,B' (default: '0,0,0')",
        "tolerance": "Color matching tolerance (0.0 to 1.0)"
    }, inherits_from='ImageUtilityNode')

    # ImageCASBatch tooltips (inherits from: ImageUtilityNode)
    TooltipManager.register_tooltips("ImageCASBatch", {
        "image": "Input image to be processed (IMAGE type)",
        "amount": "Strength of the contrast adaptive sharpening effect (0.0 to 1.0)",
        "batch_size": "Number of images to process in each batch (1 to 64)"
    }, inherits_from='ImageUtilityNode')

    # ImageScaleToTarget tooltips (inherits from: ImageUtilityNode)
    TooltipManager.register_tooltips("ImageScaleToTarget", {
        "image": "Input image to be scaled (IMAGE type)",
        "target_image": "Image whose dimensions will be used as the target size (IMAGE type)",
        "upscale_method": "Method used for upscaling ('nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos')",
        "crop": "Cropping method to use if aspect ratios don't match ('disabled', 'center')"
    }, inherits_from='ImageUtilityNode')
