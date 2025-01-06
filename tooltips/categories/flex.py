"""Tooltips for flex-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for flex nodes"""

    # EffectVisualizer tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("EffectVisualizer", {
        "video_frames": "Input video frames (IMAGE type)",
        "feature": "Feature to visualize (FEATURE type)",
        "text_color": "Color for text overlay (RGB string, e.g., \"(255,255,255)\")",
        "font_scale": "Scale factor for the font size (0.1 to 2.0)"
    }, inherits_from='RyanOnTheInside')

    # ProximityVisualizer tooltips (inherits from: EffectVisualizer)
    TooltipManager.register_tooltips("ProximityVisualizer", {
        "anchor_locations": "Locations of anchor points (LOCATION type)",
        "query_locations": "Locations of query points (LOCATION type)",
        "feature": "Proximity feature to visualize (FEATURE type)",
        "anchor_color": "Color for anchor points (RGB string, e.g., \"(255,0,0)\")",
        "query_color": "Color for query points (RGB string, e.g., \"(0,255,0)\")",
        "line_color": "Color for the line connecting closest points (RGB string, e.g., \"(0,0,255)\")"
    }, inherits_from='EffectVisualizer')

    # PitchVisualizer tooltips (inherits from: EffectVisualizer)
    TooltipManager.register_tooltips("PitchVisualizer", {
        "video_frames": "Input video frames (IMAGE type)",
        "feature": "Pitch feature to visualize (FEATURE type)",
        "text_color": "Color of the text overlay (RGB string, e.g., \"(255,255,255)\")",
        "font_scale": "Scale factor for the font size (0.1 to 2.0)"
    }, inherits_from='EffectVisualizer')

    # FlexBase tooltips (inherits from: ABC)
    TooltipManager.register_tooltips("FlexBase", {
        "strength": """Overall strength of the effect (0.0 to 1.0)
        
Higher values create more dramatic changes, while lower values are more subtle.""",
        "feature_threshold": """Minimum feature value to trigger the effect (0.0 to 1.0)
        
Only applies the effect when the feature value exceeds this threshold.
Lower values make the effect more sensitive, higher values make it more selective.""",
        "feature_param": """Choose which parameter of the effect to modulate
        
Each effect type has different parameters you can control.
Hover over each option to see what it does.""",
        "feature_mode": """How to apply the feature modulation:
- relative: Changes are centered around the original value
- absolute: Changes scale directly from zero to the maximum""",
        "opt_feature": """Optional feature input for modulation
        
Connect any feature node here to control the effect.
Features can come from audio, motion, color, or other sources."""
    }, inherits_from='ABC', description="""Use features from audio, MIDI, motion, or other sources to dynamically control and animate node parameters.
    
Features can modulate a variety of parameters to create dynamic effects that respond to your content.
Great for creating music videos, reactive animations, or automated effects.""")

    # FlexExpressionEditor tooltips (inherits from: ExpressionEditor)
    TooltipManager.register_tooltips("FlexExpressionEditor", {
        "feature": "Feature used to modulate the effect (FEATURE type)",
        "feature_pipe": "Feature pipe containing frame information (FEATURE_PIPE type)",
        "feature_threshold": "Threshold for feature activation (0.0 to 1.0)"
    }, inherits_from='ExpressionEditor', description="Use flex-features to control the facial  expressions. Click the '?' icon to find tutorial video on YouTube.")

    # ManualFeaturePipe tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("ManualFeaturePipe", {
        "frame_rate": "Frame rate of the video (1.0 to 120.0 fps)",
        "frame_count": "Total number of frames (minimum: 1)",
        "width": "Width of the output feature (64 to 4096)",
        "height": "Height of the output feature (64 to 4096)"
    }, inherits_from='RyanOnTheInside')

    # FeatureModulationBase tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("FeatureModulationBase", {
        "feature": "Input feature to be processed",
        "feature_pipe": "Feature pipe for frame synchronization"
    }, inherits_from='RyanOnTheInside', description="Process and modify features to create custom animation curves. Combine multiple features, adjust timing, or transform values to achieve desired effects.")

    # ProcessedFeature tooltips (inherits from: type)
    TooltipManager.register_tooltips("ProcessedFeature", {
        # TODO: Add parameter tooltips
    }, inherits_from='type')

    # FeatureMixer tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureMixer", {
        "base_gain": "Overall amplification of the feature values (0.0 to 10.0)",
        "floor": "Minimum value for the processed feature (0.0 to 1.0)",
        "ceiling": "Maximum value for the processed feature (0.0 to 10.0)",
        "peak_sharpness": "Sharpness of peaks in the feature curve (0.1 to 10.0)",
        "valley_sharpness": "Sharpness of valleys in the feature curve (0.1 to 10.0)",
        "attack": "Speed at which the envelope follower responds to increasing values (0.01 to 1.0)",
        "release": "Speed at which the envelope follower responds to decreasing values (0.01 to 1.0)",
        "smoothing": "Amount of smoothing applied to the final curve (0.0 to 1.0)"
    }, inherits_from='FeatureModulationBase')

    # FeatureScaler tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureScaler", {
        "scale_type": "Type of scaling to apply (\"linear\", \"logarithmic\", \"exponential\", \"inverse\")",
        "min_output": "Minimum output value after scaling (0.0 to 1.0)",
        "max_output": "Maximum output value after scaling (0.0 to 1.0)",
        "exponent": "Exponent for exponential scaling (0.1 to 10.0)"
    }, inherits_from='FeatureModulationBase')

    # FeatureCombine tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureCombine", {
        "feature1": "First input feature",
        "feature2": "Second input feature",
        "operation": "Mathematical operation to perform (\"add\", \"subtract\", \"multiply\", \"divide\", \"max\", \"min\")",
        "weight1": "Weight applied to feature1 (0.0 to 1.0)",
        "weight2": "Weight applied to feature2 (0.0 to 1.0)"
    }, inherits_from='FeatureModulationBase')

    # FeatureMath tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureMath", {
        "y": "Input value",
        "operation": "Mathematical operation to perform (\"add\", \"subtract\", \"multiply\", \"divide\", \"max\", \"min\")"
    }, inherits_from='FeatureModulationBase')

    # FeatureSmoothing tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureSmoothing", {
        "smoothing_type": "Type of smoothing to apply (\"moving_average\", \"exponential\", \"gaussian\")",
        "window_size": "Size of the smoothing window for moving average and gaussian (3 to 21, odd numbers only)",
        "alpha": "Smoothing factor for exponential smoothing (0.0 to 1.0)",
        "sigma": "Standard deviation for gaussian smoothing (0.1 to 5.0)"
    }, inherits_from='FeatureModulationBase')

    # FeatureOscillator tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureOscillator", {
        "oscillator_type": "Type of oscillation (\"sine\", \"square\", \"sawtooth\", \"triangle\")",
        "frequency": "Frequency of oscillation (0.1 to 10.0)",
        "amplitude": "Amplitude of oscillation (0.0 to 1.0)",
        "phase_shift": "Phase shift of oscillation (0.0 to 2π)",
        "blend": "Blend factor between original feature and oscillation (0.0 to 1.0)"
    }, inherits_from='FeatureModulationBase')

    # FeatureFade tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureFade", {
        "feature1": "First input feature to fade from",
        "feature2": "Second input feature to fade to",
        "fader": "Static fader value to control the blend (0.0 to 1.0)",
        "control_feature": "Optional feature to dynamically control the fader value"
    }, inherits_from='FeatureModulationBase')

    # FeatureRebase tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureRebase", {
        "lower_threshold": "Lower threshold for feature values (0.0 to 1.0)",
        "upper_threshold": "Upper threshold for feature values (0.0 to 1.0)",
        "invert_output": "Whether to invert the output feature values"
    }, inherits_from='FeatureModulationBase')

    # FeatureRenormalize tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureRenormalize", {
        "feature": "Input feature to be renormalized",
        "lower_threshold": "Minimum value for the normalized output (-10000.0 to 10000.0)",
        "upper_threshold": "Maximum value for the normalized output (-10000.0 to 10000.0)"
    }, inherits_from='FeatureModulationBase')

    # PreviewFeature tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("PreviewFeature", {
        "feature": "Input feature to preview"
    }, inherits_from='FeatureModulationBase')

    # FeatureTruncateOrExtend tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureTruncateOrExtend", {
        "feature": "Input feature to truncate or extend",
        "target_feature_pipe": "Feature pipe to match length with",
        "fill_method": "Method to fill extended frames ('zeros', 'ones', 'average', 'random', 'repeat')"
    }, inherits_from='FeatureModulationBase')

    # FeatureAccumulate tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureAccumulate", {
        "feature": "Input feature to accumulate",
        "start": "Starting value for normalized output (0.0 to 1.0)",
        "end": "Ending value for normalized output (0.0 to 1.0)",
        "threshold": "Minimum value to consider for accumulation (0.0 to 1.0)",
        "skip_thresholded": "Whether to skip values below threshold",
        "frames_window": "Number of frames to process at once (0 for all frames)",
        "deccumulate": "Whether to alternate accumulation direction between windows"
    }, inherits_from='FeatureModulationBase')

    # FeatureContiguousInterpolate tooltips (inherits from: FeatureModulationBase)
    TooltipManager.register_tooltips("FeatureContiguousInterpolate", {
        "feature": "Input feature to interpolate",
        "threshold": "Threshold for identifying contiguous segments (0.0 to 1.0)",
        "start": "Starting value for interpolation (0.0 to 1.0)",
        "end": "Ending value for interpolation (0.0 to 1.0)",
        "easing": "Easing function for interpolation",
        "fade_out": "Number of frames for fade-out after each segment (0 to 100)"
    }, inherits_from='FeatureModulationBase')

    # ProximityFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("ProximityFeatureNode", {
        "anchor_locations": "Reference locations for proximity calculation",
        "query_locations": "Locations to calculate proximity from anchors",
        "normalization_method": "Method to normalize proximity values ('frame' or 'minmax')"
    }, inherits_from='FeatureExtractorBase')

    # ProximityFeatureInput tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("ProximityFeatureInput", {
        # Base class with no direct parameters
    }, inherits_from='RyanOnTheInside', description="Create and manipulate location data for proximity-based effects. Track distances between points, analyze spatial relationships, and generate position-based animations.")

    # LocationFromMask tooltips (inherits from: ProximityFeatureInput)
    TooltipManager.register_tooltips("LocationFromMask", {
        "masks": "Input masks to extract locations from",
        "method": "Method to extract locations ('mask_boundary' or 'mask_center')",
        "depth_maps": "Optional depth maps for z-coordinate extraction"
    }, inherits_from='ProximityFeatureInput')

    # LocationFromPoint tooltips (inherits from: ProximityFeatureInput)
    TooltipManager.register_tooltips("LocationFromPoint", {
        "x": "X-coordinate of the point (0.0 or greater)",
        "y": "Y-coordinate of the point (0.0 or greater)",
        "z": "Z-coordinate of the point (0.0 to 1.0)",
        "batch_count": "Number of copies to generate (minimum: 1)"
    }, inherits_from='ProximityFeatureInput')

    # LocationTransform tooltips (inherits from: ProximityFeatureInput)
    TooltipManager.register_tooltips("LocationTransform", {
        "locations": "Input locations to transform",
        "feature": "Feature to modulate the transformation",
        "transformation_type": "Type of transformation ('translate' or 'scale')",
        "transformation_value": "Scale factor for the transformation (default: 1.0)"
    }, inherits_from='ProximityFeatureInput')

    # MIDILoadAndExtract tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("MIDILoadAndExtract", {
        "midi_file": "MIDI file to load from the midi_files directory",
        "extraction_method": """Choose what MIDI information to analyze:

- Velocity: How hard notes are played - great for impact-based effects
- Pitch: Which notes are being played - good for melody-following effects
- Note On/Off: When notes start/stop - perfect for precise timing
- Note Duration: How long notes are held - use for sustained effects
- Note Density: How many notes are playing - good for complexity-based effects
- Pitchbend: Pitch wheel movements - great for smooth transitions
- Aftertouch: Key pressure changes - useful for pressure-sensitive effects
- Poly Pressure: Per-note pressure - detailed expression control
- Modulation (CC1): Mod wheel - typically used for vibrato/intensity
- Breath (CC2): Breath controller - good for wind instrument effects
- Foot Controller (CC4): Foot pedal - useful for gradual changes
- Volume (CC7): Channel volume - overall level control
- Balance (CC8): Stereo balance - left/right positioning
- Pan (CC10): Stereo panning - spatial movement
- Expression (CC11): Expression control - dynamic volume changes
- Sustain (CC64): Sustain pedal - on/off state""",
        "track_selection": "Track to analyze ('all' or specific track number)",
        "chord_only": "When true, only considers full chords (default: false)",
        "notes": "Comma-separated list of MIDI note numbers (default: empty)"
    }, inherits_from='FeatureExtractorBase')

    # AudioFeatureExtractor tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("AudioFeatureExtractor", {
        "audio": "Input audio to analyze",
        "extraction_method": """Choose how to analyze the audio:

- amplitude_envelope: Overall loudness changes - great for syncing with dramatic moments
- rms_energy: Continuous energy level - smoother than amplitude, good for sustained effects
- spectral_centroid: Brightness of the sound - high for sharp/crisp sounds, low for bass/warm sounds
- onset_strength: Detects new sounds/beats - perfect for rhythmic effects
- chroma_features: Musical note content - useful for harmony-based effects"""
    }, inherits_from='FeatureExtractorBase')

    # AudioFeatureExtractorFirst tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("AudioFeatureExtractorFirst", {
        "audio": "Input audio to analyze",
        "extraction_method": """Choose how to analyze the audio:

- amplitude_envelope: Overall loudness changes - great for syncing with dramatic moments
- rms_energy: Continuous energy level - smoother than amplitude, good for sustained effects
- spectral_centroid: Brightness of the sound - high for sharp/crisp sounds, low for bass/warm sounds
- onset_strength: Detects new sounds/beats - perfect for rhythmic effects
- chroma_features: Musical note content - useful for harmony-based effects"""
    }, inherits_from='FeatureExtractorBase')

    # RhythmFeatureExtractor tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("RhythmFeatureExtractor", {
        "audio": "Input audio to analyze",
        "extraction_method": """Choose how to analyze the rhythm:

- beat_locations: Marks exact beat timings - perfect for precise beat-synced effects
- tempo: Overall speed of the music - use for speed-based animations
- onset_strength: How strong each beat is - great for impact-based effects
- beat_emphasis: Highlights stronger beats - good for accenting main beats
- syncopation: Detects off-beat rhythms - interesting for complex animations
- rhythm_regularity: How steady the rhythm is - use for stability-based effects
- down_beats: Marks main beats (1st beat) - strong rhythmic emphasis
- up_beats: Marks other beats - lighter rhythmic emphasis""",
        "time_signature": "Number of beats per measure (e.g., 4 for 4/4 time)"
    }, inherits_from='FeatureExtractorBase')

    # PitchFeatureExtractor tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("PitchFeatureExtractor", {
        "audio": "Input audio to analyze",
        "extraction_method": """Choose how to analyze the pitch:

- frequency: Raw pitch frequency - good for following melody
- semitone: Rounds to nearest musical note - better for musical effects
- pitch_direction: Whether pitch is rising or falling - great for directional animations
- vibrato_signal: Detects pitch wobble - perfect for tremolo effects
- vibrato_strength: How intense the vibrato is - use for intensity-based effects""",
        "opt_crepe_model": """CREPE model size for pitch detection:
- none: Use basic pitch detection
- tiny/small: Fast but less accurate
- medium: Good balance of speed and accuracy
- large/full: Most accurate but slower""",
        "opt_pitch_range_collections": "Optional collections of pitch ranges to consider"
    }, inherits_from='FeatureExtractorBase')

    # PitchAbstraction tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("PitchAbstraction", {
        # Base class with no direct parameters
    }, inherits_from='RyanOnTheInside', description="Work with musical pitch data to create audio-reactive effects. Define pitch ranges, detect specific notes, and create animations based on vocal or instrumental frequencies.")

    # PitchRangeNode tooltips (inherits from: PitchAbstraction)
    TooltipManager.register_tooltips("PitchRangeNode", {
        "min_pitch": "Minimum frequency of the pitch range in Hz (20.0 to 2000.0)",
        "max_pitch": "Maximum frequency of the pitch range in Hz (20.0 to 2000.0)",
        "previous_range_collection": "Optional previous pitch range collection to append to"
    }, inherits_from='PitchAbstraction')

    # PitchRangePresetNode tooltips (inherits from: PitchAbstraction)
    TooltipManager.register_tooltips("PitchRangePresetNode", {
        "preset": "Preset vocal range to use ('Bass', 'Baritone', 'Tenor', 'Alto', 'Mezzo-soprano', 'Soprano', 'Contralto')",
        "previous_range_collection": "Optional previous pitch range collection to append to"
    }, inherits_from='PitchAbstraction')

    # PitchRangeByNoteNode tooltips (inherits from: PitchAbstraction)
    TooltipManager.register_tooltips("PitchRangeByNoteNode", {
        "chord_only": "If true, only detects when all specified notes are present simultaneously",
        "pitch_tolerance_percent": "Tolerance percentage for pitch detection (0.0 to 100.0)",
        "notes": "Comma-separated list of MIDI note numbers",
        "previous_range_collection": "Optional previous pitch range collection to append to"
    }, inherits_from='PitchAbstraction')

    # FeatureExtractorBase tooltips (inherits from: RyanOnTheInside, ABC)
    TooltipManager.register_tooltips("FeatureExtractorBase", {
        "extraction_method": "Method used to extract features",
        "frame_rate": "Frame rate of the video (1.0 to 120.0 fps)",
        "frame_count": "Total number of frames (minimum: 1)",
        "width": "Width of the output feature (64 to 4096)",
        "height": "Height of the output feature (64 to 4096)"
    }, inherits_from=['RyanOnTheInside', 'ABC'], description="Feature extractors allow you to extract animation data from various sources like audio, video, or manual input. Use this with the FlexFeature compatible nodes to convert real-world information into features that can control effects and animations.")

    # ManualFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("ManualFeatureNode", {
        "frame_numbers": "Comma-separated list of frame numbers (e.g., \"0,10,20\")",
        "values": "Comma-separated list of values (e.g., \"0.0,0.5,1.0\")",
        "last_value": "Value for the last frame (default: 1.0)",
        "interpolation_method": "Method for interpolating between values ('none', 'linear', 'ease_in', 'ease_out')"
    }, inherits_from='FeatureExtractorBase')

    # ManualFeatureFromPipe tooltips (inherits from: ManualFeatureNode)
    TooltipManager.register_tooltips("ManualFeatureFromPipe", {
        "feature_pipe": "Input feature pipe",
        "frame_numbers": "Comma-separated list of frame numbers (e.g., \"0,10,20\")",
        "values": "Comma-separated list of values (e.g., \"0.0,0.5,1.0\")",
        "last_value": "Value for the last frame (default: 1.0)"
    }, inherits_from='ManualFeatureNode')

    # TimeFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("TimeFeatureNode", {
        "speed": "How quickly the effect progresses (0.1 to 10.0)",
        "offset": "Shifts the starting point of the effect (0.0 to 1.0)"
    }, inherits_from='FeatureExtractorBase')

    # DepthFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("DepthFeatureNode", {
        "depth_maps": "Input depth maps to analyze",
        "extraction_method": """Choose how to analyze the depth information:

- mean_depth: Overall depth of the scene - higher when objects are further away
- depth_variance: How much depth varies in the scene - higher for complex 3D scenes
- depth_range: Distance between closest and farthest points
- gradient_magnitude: How sharply depth changes - high for edges and steep surfaces
- foreground_ratio: Percentage of scene in the front layer
- midground_ratio: Percentage of scene in the middle layer
- background_ratio: Percentage of scene in the back layer"""
    }, inherits_from='FeatureExtractorBase')

    # ColorFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("ColorFeatureNode", {
        "images": "Input images to analyze",
        "extraction_method": """Choose how to analyze the colors:

- dominant_color: Most prominent color in the scene
- color_variance: How much colors vary - higher for colorful scenes
- saturation: Overall color intensity
- red_ratio: Amount of red in the scene
- green_ratio: Amount of green in the scene
- blue_ratio: Amount of blue in the scene"""
    }, inherits_from='FeatureExtractorBase')

    # BrightnessFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("BrightnessFeatureNode", {
        "images": "Input images to analyze",
        "extraction_method": """Choose how to analyze brightness:

- mean_brightness: Overall brightness of the scene
- brightness_variance: How much brightness varies - high for high contrast scenes
- brightness_histogram: Distribution of brightness values
- dark_ratio: Percentage of dark areas in the scene
- mid_ratio: Percentage of medium-bright areas
- bright_ratio: Percentage of bright areas"""
    }, inherits_from='FeatureExtractorBase')

    # MotionFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("MotionFeatureNode", {
        "images": "Input images to analyze",
        "extraction_method": """Choose how to analyze motion:

- mean_motion: Overall amount of movement in the scene
- max_motion: Speed of the fastest moving parts
- motion_direction: Main direction of movement
- horizontal_motion: Amount of left-right movement
- vertical_motion: Amount of up-down movement
- motion_complexity: How chaotic or varied the motion is
- motion_speed: Speed of movement adjusted for frame rate""",
        "flow_method": "Method for calculating optical flow ('Farneback', 'LucasKanade', 'PyramidalLK')",
        "flow_threshold": "Minimum motion magnitude to consider (0.0 to 10.0)",
        "magnitude_threshold": "Relative threshold for motion magnitude (0.0 to 1.0)"
    }, inherits_from='FeatureExtractorBase')

    # AreaFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("AreaFeatureNode", {
        "masks": "Input mask sequence to analyze",
        "extraction_method": """Choose how to analyze the masked area:

- total_area: Total size of the masked region - good for tracking overall coverage
- largest_contour: Size of the biggest connected region - useful for tracking main objects
- bounding_box: Area of the rectangle containing the mask - great for object size changes""",
        "threshold": "Threshold value for considering pixels as part of the area (0.0 to 1.0)"
    }, inherits_from='FeatureExtractorBase')

    # DrawableFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("DrawableFeatureNode", {
        "points": "JSON string of [frame, value] pairs",
        "extraction_method": """Choose how to analyze drawn points:

- drawn: Creates a feature from manually specified points - perfect for custom animations""",
        "min_value": "Minimum value for the feature (-100.0 to 100.0)",
        "max_value": "Maximum value for the feature (-100.0 to 100.0)",
        "interpolation_method": """Method for interpolating between points:
- linear: Smooth straight-line transitions
- cubic: Smooth curved transitions
- nearest: Snap to closest point
- zero: No interpolation, instant changes
- hold: Keep previous value until next point
- ease_in: Gradually accelerate changes
- ease_out: Gradually decelerate changes""",
        "fill_value": "Value to use for undefined regions (-100.0 to 100.0)"
    }, inherits_from='FeatureExtractorBase')

    # FlexExternalModulator tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("FlexExternalModulator", {
        # Base class with no direct parameters
    }, inherits_from='RyanOnTheInside')

    # FeatureToWeightsStrategy tooltips (inherits from: FlexExternalModulator)
    TooltipManager.register_tooltips("FeatureToWeightsStrategy", {
        "feature": "Input feature to be converted to weights"
    }, inherits_from='FlexExternalModulator')

    # FeatureToSplineData tooltips (inherits from: FlexExternalModulator)
    TooltipManager.register_tooltips("FeatureToSplineData", {
        "feature": "Input feature to be converted to spline data",
        "mask_width": "Width of the output mask (8 to 4096)",
        "mask_height": "Height of the output mask (8 to 4096)",
        "sampling_method": "Method for sampling points ('path', 'time', 'controlpoints')",
        "interpolation": "Spline interpolation method ('cardinal', 'monotone', 'basis', 'linear', 'step-before', 'step-after', 'polar', 'polar-reverse')",
        "tension": "Tension parameter for cardinal splines (0.0 to 1.0)",
        "repeat_output": "Number of times to repeat the output (1 to 4096)",
        "float_output_type": "Type of float output ('list', 'pandas series', 'tensor')",
        "min_value": "Optional minimum value for normalization",
        "max_value": "Optional maximum value for normalization"
    }, inherits_from='FlexExternalModulator')

    # SplineFeatureModulator tooltips (inherits from: FlexExternalModulator)
    TooltipManager.register_tooltips("SplineFeatureModulator", {
        "coordinates": "JSON string containing spline control points",
        "feature": "Input feature for modulation",
        "mask_width": "Width of the output mask (8 to 4096)",
        "mask_height": "Height of the output mask (8 to 4096)",
        "min_speed": "Minimum speed of movement along the spline (0.0 to 10.0)",
        "max_speed": "Maximum speed of movement along the spline (0.0 to 10.0)",
        "float_output_type": "Type of float output ('list', 'pandas series', 'tensor')",
        "min_value": "Optional minimum value for normalization",
        "max_value": "Optional maximum value for normalization"
    }, inherits_from='FlexExternalModulator')

    # SplineRhythmModulator tooltips (inherits from: FlexExternalModulator)
    TooltipManager.register_tooltips("SplineRhythmModulator", {
        "coordinates": "JSON string containing spline control points",
        "feature": "Input feature for rhythm modulation",
        "mask_width": "Width of the output mask (8 to 4096)",
        "mask_height": "Height of the output mask (8 to 4096)",
        "smoothing": "Amount of smoothing applied to the feature values (0.0 to 1.0)",
        "direction": "Direction of movement along the spline ('forward', 'backward', 'bounce')",
        "float_output_type": "Type of float output ('list', 'pandas series', 'tensor')",
        "min_value": "Optional minimum value for normalization",
        "max_value": "Optional maximum value for normalization"
    }, inherits_from='FlexExternalModulator')

    # FeatureToFloat tooltips (inherits from: FlexExternalModulator)
    TooltipManager.register_tooltips("FeatureToFloat", {
        "feature": "Input feature to be converted to float values"
    }, inherits_from='FlexExternalModulator')

    # DepthShapeModifier tooltips (inherits from: FlexExternalModulator)
    TooltipManager.register_tooltips("DepthShapeModifier", {
        "depth_map": "Input depth map to be modified",
        "mask": "Mask defining areas to apply the shape modification",
        "gradient_steepness": "Controls the steepness of the spherical gradient (0.1 to 10.0)",
        "depth_min": "Minimum depth value for modified areas (0.0 to 1.0)",
        "depth_max": "Maximum depth value for modified areas (0.0 to 1.0)",
        "strength": "Overall strength of the modification (0.0 to 1.0)"
    }, inherits_from='FlexExternalModulator')

    # DepthShapeModifierPrecise tooltips (inherits from: FlexExternalModulator)
    TooltipManager.register_tooltips("DepthShapeModifierPrecise", {
        "depth_map": "Input depth map to be modified",
        "mask": "Mask defining areas to apply the shape modification",
        "gradient_steepness": "Controls the steepness of the spherical gradient (0.1 to 10.0)",
        "depth_min": "Minimum depth value for modified areas (0.0 to 1.0)",
        "depth_max": "Maximum depth value for modified areas (0.0 to 1.0)",
        "strength": "Overall strength of the modification (0.0 to 1.0)",
        "composite_method": "Method for compositing the modified depth ('linear', 'depth_aware', 'add', 'subtract', 'multiply', 'divide', 'screen', 'overlay', 'protrude')"
    }, inherits_from='FlexExternalModulator')

    # WhisperFeatureNode tooltips (inherits from: FeatureExtractorBase)
    TooltipManager.register_tooltips("WhisperFeatureNode", {
        "alignment_data": "JSON string containing Whisper transcription alignment data",
        "trigger_set": "Set of triggers that define value sequences based on spoken words",
        "context_size": "Number of words to consider for context (0-10)",
        "overlap_mode": "How to handle overlapping triggers: blend (smooth transition), replace (use latest), or add (combine values)",
        "feature_name": "Type of feature to extract: word_timing (peaks at words), segment_timing (plateaus during segments), trigger_values (word-based values), speech_density (words/sec), silence_ratio (speech vs silence)"
    }, inherits_from='FeatureExtractorBase', description="Extract features from Whisper speech transcription data. Create timing, trigger, and density features based on speech patterns.")

    # TriggerBuilder tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("TriggerBuilder", {
        "pattern": "Word or phrase to match in the transcription",
        "start_value": "Value when the word starts (0-1)",
        "end_value": "Value when the word ends (0-1)",
        "match_mode": "How to match the pattern: exact (whole word), contains (substring), regex (patterns), phonetic (similar sounds)",
        "fade_type": "Value transition: none (instant), linear (smooth), smooth (eased)",
        "duration_frames": "Duration in frames (0 = use word duration)",
        "blend_mode": "How this trigger combines with previous ones: blend (average), add (sum), multiply (product), max (highest)",
        "previous_triggers": "Optional previous trigger set to chain with"
    }, inherits_from='RyanOnTheInside', description="Create triggers that respond to specific words or phrases in Whisper transcriptions. Can be chained together to build complex combinations.")


    # ContextModifier tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("ContextModifier", {
        "trigger_set": "Set of triggers to modify",
        "modifier_type": "Context type: timing (duration/position), sentiment (positive/negative), speaker (who's speaking), sequence (pattern in word sequence)",
        "condition": "Python expression that determines when to apply modification. Available variables depend on modifier_type",
        "value_adjust": "How much to modify trigger values (1.0 = no change, >1.0 amplify, <1.0 reduce)",
        "window_size": "Number of words to look at for context"
    }, inherits_from='RyanOnTheInside', description="Modify trigger behavior based on speech context. Can adjust values based on word timing, sentiment, speaker changes, or sequence patterns.")

    # WhisperToPromptTravel tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("WhisperToPromptTravel", {
        "segments_alignment": "JSON string of Whisper segment alignments",
        "fps": "Frame rate for converting time to frame numbers"
    }, inherits_from='RyanOnTheInside', description="Convert Whisper transcription timing to prompt travel format for text animation.")

    # WhisperTextRenderer tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("WhisperTextRenderer", {
        "images": "Input video frames",
        "feature": "Whisper feature containing alignment data",
        "font_size": "Size of the rendered text",
        "font_name": "Font to use for rendering",
        "position": "Vertical position of text",
        "horizontal_align": "Horizontal alignment of text",
        "margin": "Margin from frame edges",
        "animation_type": "Type of text animation",
        "animation_duration": "Duration of text animations",
        "max_width": "Maximum width for text wrapping",
        "bg_color": "Background color in hex format",
        "text_color": "Text color in hex format",
        "opacity": "Overall opacity of text overlay"
    }, inherits_from='RyanOnTheInside', description="Render animated text overlays from Whisper transcription with various animation styles.")

    # ManualWhisperAlignmentData tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("ManualWhisperAlignmentData", {
        "alignment_text": "JSON objects defining speech segments. Each object needs: value (text content), start (time in seconds), end (time in seconds)"
    }, inherits_from='RyanOnTheInside', description="""Creates alignment data from manually entered text and timings.
    
    **Note**: For proper whisper alignment data, you should use ComfyUI-Whisper:
    https://github.com/yuvraj108c/ComfyUI-Whisper
   
    You can enter either word alignments or segment alignments.
    Format should match ComfyUI-Whisper output.
    
""")