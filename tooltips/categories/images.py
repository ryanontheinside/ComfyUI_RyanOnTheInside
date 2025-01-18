"""Tooltips for images-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for images nodes"""

    # FlexImageBase tooltips (inherits from: RyanOnTheInside, FlexBase)
    TooltipManager.register_tooltips("FlexImageBase", {
        "images": "Input image sequence to be processed (IMAGE type)",
        "feature_param": """Choose which parameter to modulate with the input feature
        
Each node type has different parameters that can be modulated:
- 'None': No parameter modulation (default behavior)
- Other options depend on the specific node type"""
    }, inherits_from=['RyanOnTheInside', 'FlexBase'])

    # FlexImageEdgeDetect tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageEdgeDetect", {
        "low_threshold": "Lower bound for the hysteresis thresholding (0 to 255)",
        "high_threshold": "Upper bound for the hysteresis thresholding (0 to 255)",
        "feature_param": """Choose which parameter to modulate:
        
- low_threshold: Dynamically adjust edge sensitivity
- high_threshold: Dynamically adjust edge strength
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImagePosterize tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImagePosterize", {
        "max_levels": "Maximum number of color levels per channel. Higher values preserve more color detail (2 to 256)",
        "dither_strength": "Intensity of dithering effect. Higher values reduce color banding (0.0 to 1.0)",
        "channel_separation": "Degree of separation between color channels. Creates color shift effects (0.0 to 1.0)",
        "gamma": "Gamma correction applied before posterization. Affects brightness distribution (0.1 to 2.2)",
        "dither_method": """Method used for dithering:
- ordered: Fast Bayer matrix dithering, good for retro effects
- floyd: Floyd-Steinberg dithering, better for natural gradients
- none: No dithering, creates sharp color transitions""",
        "feature_param": """Choose which parameter to modulate:
        
- max_levels: Dynamically adjust color quantization
- dither_strength: Dynamically adjust dithering intensity
- channel_separation: Dynamically adjust color channel offsets
- gamma: Dynamically adjust brightness distribution
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageKaleidoscope tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageKaleidoscope", {
        "segments": "Number of mirror segments (2 to 32)",
        "center_x": "X-coordinate of the effect center (0.0 to 1.0)",
        "center_y": "Y-coordinate of the effect center (0.0 to 1.0)",
        "zoom": "Zoom factor for the effect (0.1 to 2.0)",
        "rotation": "Rotation angle of the effect (0.0 to 360.0)",
        "precession": "Rate of rotation change over time (-1.0 to 1.0)",
        "speed": "Speed of the effect animation (0.1 to 5.0)",
        "feature_param": """Choose which parameter to modulate:
        
- segments: Dynamically adjust number of mirror segments
- zoom: Dynamically adjust magnification
- rotation: Dynamically adjust pattern rotation
- precession: Dynamically adjust rotation speed
- speed: Dynamically adjust animation speed
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageColorGrade tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageColorGrade", {
        "intensity": "Strength of the color grading effect (0.0 to 1.0)",
        "mix": "Blend factor between original and graded image (0.0 to 1.0)",
        "lut_file": "Path to the LUT file",
        "feature_param": """Choose which parameter to modulate:
        
- intensity: Dynamically adjust grading strength
- mix: Dynamically adjust blend amount
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageGlitch tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageGlitch", {
        "glitch_type": """Type of glitch effect to apply:
- digital: Modern digital artifacts with RGB shifts and block displacement
- analog: Classic TV distortion with scan lines, ghosting, and vertical hold issues
- compression: JPEG-like artifacts with block corruption and quantization
- wave: Smooth wave-based distortions with multiple harmonics
- corrupt: Random data corruption with line and block artifacts""",
        "intensity": "Overall strength of the glitch effect. Higher values create more pronounced distortions (0.0 to 1.0)",
        "block_size": "Size of blocks for digital glitches and compression artifacts. Larger sizes create more visible blocks (8 to 128)",
        "wave_amplitude": "Height of wave distortions. Controls the magnitude of wave-based displacement (0.0 to 1.0)",
        "wave_frequency": "Frequency of wave patterns. Higher values create more rapid oscillations (0.1 to 20.0)",
        "corruption_amount": "Probability and intensity of corruption artifacts. Affects how often and how severely glitches occur (0.0 to 1.0)",
        "time_seed": "Seed for random glitch generation. Same seed produces consistent glitch patterns (0 to 10000)",
        "feature_param": """Choose which parameter to modulate:
        
- intensity: Dynamically adjust overall glitch strength
- block_size: Dynamically adjust artifact size
- wave_amplitude: Dynamically adjust wave distortion
- wave_frequency: Dynamically adjust wave patterns
- corruption_amount: Dynamically adjust glitch frequency
- time_seed: Dynamically change random patterns
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageChromaticAberration tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageChromaticAberration", {
        "shift_amount": "Amount of RGB channel separation. Higher values create more color fringing (0.0 to 0.5)",
        "angle": "Direction of the chromatic aberration effect in degrees. 0° is horizontal, 90° is vertical (0.0 to 720.0)",
        "feature_param": """Choose which parameter to modulate:
        
- shift_amount: Dynamically adjust color separation
- angle: Dynamically adjust separation direction
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImagePixelate tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImagePixelate", {
        "pixel_size": "Size of each pixelated block. Larger values create more pronounced pixelation (1 to 100)",
        "feature_param": """Choose which parameter to modulate:
        
- pixel_size: Dynamically adjust pixelation size
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageBloom tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageBloom", {
        "threshold": "Brightness threshold for bloom effect. Only pixels brighter than this value will glow (0.0 to 1.0)",
        "blur_amount": "Base radius of the bloom effect. Higher values create softer, more diffused glow (0.0 to 50.0)",
        "intensity": "Overall strength of the bloom effect. Controls the brightness of the glow (0.0 to 1.0)",
        "num_passes": "Number of bloom passes. More passes create layered, atmospheric glow effects (1 to 8)",
        "color_bleeding": "Amount of color spreading between bright areas. Higher values create more color mixing (0.0 to 1.0)",
        "falloff": "How quickly the bloom effect diminishes with distance. Higher values create tighter, more focused glow (0.1 to 3.0)",
        "surface_scatter": "How much the bloom follows surface geometry when using normal map. Affects glow distribution (0.0 to 1.0)",
        "normal_influence": "Strength of normal map's influence on bloom direction. Controls surface-aware lighting (0.0 to 1.0)",
        "opt_normal_map": "Optional normal map for surface-aware bloom. Enables realistic light scattering based on surface geometry",
        "feature_param": """Choose which parameter to modulate:
        
- threshold: Dynamically adjust bloom threshold
- blur_amount: Dynamically adjust bloom radius
- intensity: Dynamically adjust bloom strength
- num_passes: Dynamically adjust bloom quality
- color_bleeding: Dynamically adjust color spread
- falloff: Dynamically adjust bloom falloff
- surface_scatter: Dynamically adjust surface influence
- normal_influence: Dynamically adjust normal map impact
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageTiltShift tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageTiltShift", {
        "blur_amount": "Amount of blur applied to out-of-focus areas. Higher values create stronger depth-of-field effect (0.0 to 50.0)",
        "focus_position_x": "Horizontal position of focus center. 0 = left edge, 1 = right edge (0.0 to 1.0)",
        "focus_position_y": "Vertical position of focus center. 0 = top edge, 1 = bottom edge (0.0 to 1.0)",
        "focus_width": "Width of the in-focus area relative to image width. Larger values keep more of the image sharp (0.0 to 1.0)",
        "focus_height": "Height of the in-focus area relative to image height. Larger values keep more of the image sharp (0.0 to 1.0)",
        "focus_shape": """Shape of the focus area:
- rectangle: Sharp-edged focus region with clear boundaries
- ellipse: Smooth oval focus region for natural transitions
- gradient: Continuous focus falloff for realistic depth simulation""",
        "bokeh_shape": """Shape of out-of-focus highlights:
- circular: Natural, round bokeh like modern lenses
- hexagonal: Six-sided bokeh typical of vintage lenses
- star: Decorative star-shaped bokeh for artistic effect""",
        "bokeh_size": "Size of bokeh highlights in out-of-focus areas. Larger values create more pronounced bokeh effects (0.1 to 2.0)",
        "bokeh_brightness": "Brightness multiplier for bokeh highlights. Higher values make bright points more prominent (0.5 to 2.0)",
        "chromatic_aberration": "Amount of color fringing in out-of-focus areas. Simulates lens dispersion effects (0.0 to 1.0)",
        "feature_param": """Choose which parameter to modulate:
        
- blur_amount: Dynamically adjust blur intensity
- focus_position_x: Dynamically adjust horizontal focus
- focus_position_y: Dynamically adjust vertical focus
- focus_width: Dynamically adjust focus area width
- focus_height: Dynamically adjust focus area height
- bokeh_size: Dynamically adjust bokeh highlight size
- bokeh_brightness: Dynamically adjust bokeh intensity
- chromatic_aberration: Dynamically adjust color fringing
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageParallax tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageParallax", {
        "shift_x": "Horizontal shift factor for the parallax effect (-1.0 to 1.0)",
        "shift_y": "Vertical shift factor for the parallax effect (-1.0 to 1.0)",
        "shift_z": "Z-axis shift factor for the parallax effect (-1.0 to 1.0)",
        "depth_map": "Optional depth map for 3D parallax effect",
        "feature_param": """Choose which parameter to modulate:
        
- shift_x: Dynamically adjust horizontal movement
- shift_y: Dynamically adjust vertical movement
- shift_z: Dynamically adjust depth movement
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageContrast tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageContrast", {
        "contrast": "Controls the amount of contrast adjustment (0.0 to 3.0)",
        "brightness": "Adjusts the overall brightness of the image (-1.0 to 1.0)",
        "preserve_luminosity": "When enabled, maintains the overall luminosity of the image",
        "feature_param": """Choose which parameter to modulate:
        
- contrast: Dynamically adjust contrast level
- brightness: Dynamically adjust brightness level
- preserve_luminosity: Dynamically toggle luminosity preservation
- None: No parameter modulation"""
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
        "warp_seed": "Optional seed for noise generation",
        "feature_param": """Choose which parameter to modulate:
        
- warp_strength: Dynamically adjust distortion amount
- center_x: Dynamically adjust horizontal center
- center_y: Dynamically adjust vertical center
- radius: Dynamically adjust effect radius
- warp_frequency: Dynamically adjust noise/pattern frequency
- warp_octaves: Dynamically adjust noise detail
- warp_seed: Dynamically change noise pattern
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageVignette tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageVignette", {
        "intensity": "Strength of the vignette effect (0.0 to 1.0)",
        "radius": "Radius of the vignette (0.1 to 2.0)",
        "feather": "Amount of feathering on the vignette edge (0.0 to 1.0)",
        "center_x": "X-coordinate of the vignette center (0.0 to 1.0)",
        "center_y": "Y-coordinate of the vignette center (0.0 to 1.0)",
        "feature_param": """Choose which parameter to modulate:
        
- intensity: Dynamically adjust vignette strength
- radius: Dynamically adjust vignette size
- feather: Dynamically adjust edge softness
- center_x: Dynamically adjust horizontal center
- center_y: Dynamically adjust vertical center
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageTransform tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageTransform", {
        "transform_type": "Type of transformation ('translate', 'rotate', 'scale')",
        "x_value": "X-axis transformation value (-1000.0 to 1000.0)",
        "y_value": "Y-axis transformation value (-1000.0 to 1000.0)",
        "feature_param": """Choose which parameter to modulate:
        
- x_value: Dynamically adjust horizontal transform
- y_value: Dynamically adjust vertical transform
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageHueShift tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageHueShift", {
        "hue_shift": "Amount of hue shift to apply (0.0 to 360.0)",
        "opt_mask": "Optional mask to apply the effect selectively",
        "feature_param": """Choose which parameter to modulate:
        
- hue_shift: Dynamically adjust color rotation
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageDepthWarp tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageDepthWarp", {
        "warp_strength": "Strength of the warping effect (-10.0 to 10.0)",
        "depth_map": "Depth map for warping the image",
        "feature_param": """Choose which parameter to modulate:
        
- warp_strength: Dynamically adjust depth-based distortion
- None: No parameter modulation"""
    }, inherits_from='FlexImageBase')

    # FlexImageHorizontalToVertical tooltips (inherits from: FlexImageBase)
    TooltipManager.register_tooltips("FlexImageHorizontalToVertical", {
        "blur_amount": "Amount of blur for background when using blur effect (0.1 to 100.0)",
        "background_type": "Type of background effect ('blur', 'border', 'mirror', 'gradient', 'pixelate', 'waves')",
        "border_color": "Color of the border when using border background type ('black', 'white')",
        "scale_factor": "Scale factor for the main image (0.1 to 2.0)",
        "effect_strength": "Intensity of the background effect (0.0 to 2.0)",
        "feature_param": """Choose which parameter to modulate:
        
- blur_amount: Dynamically adjust background blur
- scale_factor: Dynamically adjust image scaling
- effect_strength: Dynamically adjust effect intensity
- None: No parameter modulation"""
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
