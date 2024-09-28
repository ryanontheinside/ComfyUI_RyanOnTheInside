

NODE_CONFIGS = {}


#NOTE: this abstraction allows for the documentation to be both centrally managed and inherited
from abc import ABCMeta
class NodeConfigMeta(type):
    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        if name in NODE_CONFIGS:
            for key, value in NODE_CONFIGS[name].items():
                setattr(new_class, key, value)
        return new_class

class CombinedMeta(NodeConfigMeta, ABCMeta):
    pass

def add_node_config(node_name, config):
    NODE_CONFIGS[node_name] = config



add_node_config("MaskBase", {
    "BASE_DESCRIPTION": """
##Parameters
- `Masks`: Input mask or sequence of masks to be processed. (you can pass in a blank mask if you want)
- `Strength`: Controls the intensity of the effect (0.0 to 1.0). Higher values make the mask operation more pronounced.
- `Invert`: When enabled, reverses the mask, turning black areas white and vice versa.
- `Subtract Original`: Removes a portion of the original mask from the result (0.0 to 1.0). Higher values create more pronounced edge effects.
- `Grow with Blur`: Expands the mask edges (0.0 to 10.0). Higher values create softer, more expanded edges.
"""
})

add_node_config("TemporalMaskBase", {
    "ADDITIONAL_INFO": """
- `Start Frame`: The frame number where the effect begins (0 to 1000).
- `End Frame`: The frame number where the effect ends (0 to 1000). If set to 0, continues until the last frame.
- `Effect Duration`: Number of frames over which the effect is applied (0 to 1000). If 0, uses (End Frame - Start Frame).
- `Temporal Easing`: Controls how the effect strength changes over time, affecting the smoothness of transitions.
  - Options: "ease_in_out", "linear", "bounce", "elastic", "none"
- `Palindrome`: When enabled, the effect plays forward then reverses within the specified duration, creating a back-and-forth motion.
"""
})

add_node_config("ParticleSystemMaskBase", {
    "ADDITIONAL_INFO": """
- `particle_count`: Total number of particles in the system (1 to 10000). More particles create denser effects.
- `particle_lifetime`: How long each particle exists in seconds (0.1 to 10.0). Longer lifetimes create more persistent effects.
- `wind_strength`: Power of the wind effect (-100.0 to 100.0). Positive values blow right, negative left.
- `wind_direction`: Angle of the wind in degrees (0.0 to 360.0). 0 is right, 90 is up, etc.
- `gravity`: Strength of downward pull (-1000.0 to 1000.0). Negative values make particles float up.
- `start_frame`: Frame to begin the particle effect (0 to 1000).
- `end_frame`: Frame to stop the particle effect (0 to 1000).
- `respect_mask_boundary`: When enabled, particles stay within the mask's shape.

Optional inputs:
- `emitters`: Particle emitter configurations (PARTICLE_EMITTER type). Define where particles originate.
- `vortices`: Optional vortex configurations (VORTEX type). Create swirling effects.
- `wells`: Optional gravity well configurations (GRAVITY_WELL type). Create areas that attract or repel particles.
- `well_strength_multiplier`: Amplifies the power of gravity wells (0.0 to 10.0). Higher values create stronger attraction/repulsion.
"""
})

add_node_config("ParticleEmissionMask", {
    "TOP_DESCRIPTION": "This is the main node for particle simulations. It creates dynamic, fluid-like effects through particle simulation. Supports multiple particle emitters, force fields (Gravity Well, Vortex), and allows for complex particle behaviors including boundary-respecting particles and static body interactions.",
    "ADDITIONAL_INFO": """
- `emission_strength`: Strength of particle emission effect (0.0 to 1.0), basically opacity
- `draw_modifiers`: Visibility of vortices and gravity wells (0.0 to 1.0)
"""
})

add_node_config("OpticalFlowMaskBase", {
    "ADDITIONAL_INFO": """
- `Images`: Input image sequence for optical flow calculation
- `Masks`: Input mask sequence to be processed
- `Strength`: Overall intensity of the effect (0.0 to 1.0). Higher values create more pronounced motion-based effects.
- `Flow Method`: Technique used for optical flow calculation. Each method has different speed/accuracy tradeoffs.
  - Options: "Farneback", "LucasKanade", "PyramidalLK"
- `Flow Threshold`: Minimum motion required to trigger the effect (0.0 to 1.0). Higher values ignore subtle movements.
- `Magnitude Threshold`: Relative threshold for flow magnitude (0.0 to 1.0). Higher values focus on areas of stronger motion.
"""
})

add_node_config("OpticalFlowMaskModulation", {
    "TOP_DESCRIPTION": "This is currently the main Optical Flow node. Use it to make motion based effects.",
    "ADDITIONAL_INFO": """
- `Modulation Strength`: Intensity of the modulation effect (0.0 to 5.0). Higher values create more pronounced motion trails.
- `Blur Radius`: Smoothing applied to the flow magnitude (0 to 20 pixels). Larger values create smoother trails.
- `Trail Length`: Number of frames for the trail effect (1 to 20). Longer trails last longer.
- `Decay Factor`: Rate of trail decay over time (0.1 to 1.0). Lower values make trails fade faster.
- `Decay Style`: Method of trail decay.
  - Options: "fade" (opacity reduction), "thickness" (width reduction)
- `Max Thickness`: Maximum trail thickness for thickness-based decay (1 to 50 pixels). Larger values create thicker trails.
"""
})

add_node_config("OpticalFlowDirectionMask", {
    "TOP_DESCRIPTION": "***WORK IN PROGRESS***"
})

add_node_config("OpticalFlowParticleSystem", {
    "TOP_DESCRIPTION": "***WORK IN PROGRESS***"
})

add_node_config("MaskTransform", {
    "TOP_DESCRIPTION": "Applies geometric transformations to the mask over time.",
    "ADDITIONAL_INFO": """
- `Transform Type`: The type of transformation to apply.
  - Options: "translate", "rotate", "scale"
- `X Value`: Horizontal component of the transformation (-1000 to 1000). Positive values move right, negative left.
- `Y Value`: Vertical component of the transformation (-1000 to 1000). Positive values move up, negative down.
"""
})

add_node_config("MaskMorph", {
    "TOP_DESCRIPTION": "Applies morphological operations to the mask, changing its shape over time.",
    "ADDITIONAL_INFO": """
- `Morph Type`: The type of morphological operation to apply.
  - Options: "erode", "dilate", "open", "close"
- `Max Kernel Size`: Maximum size of the morphological kernel (3 to 21, odd numbers only). Larger values create more pronounced effects.
- `Max Iterations`: Maximum number of times to apply the operation (1 to 50). More iterations create more extreme effects.
"""
})

add_node_config("MaskMath", {
    "TOP_DESCRIPTION": "Combines two masks using various mathematical operations.",
    "ADDITIONAL_INFO": """
- `Mask B`: Second mask to combine with the input mask.
- `Combination Method`: Mathematical operation to apply.
  - Options: "add", "subtract", "multiply", "minimum", "maximum"
"""
})

add_node_config("MaskRings", {
    "TOP_DESCRIPTION": "Creates concentric ring patterns based on the distance from the mask edges.",
    "ADDITIONAL_INFO": """
- `Num Rings`: Number of rings to generate (1 to 50). More rings create more detailed patterns.
- `Max Ring Width`: Maximum width of each ring as a fraction of the total distance (0.01 to 0.5). Larger values create wider rings.
"""
})

add_node_config("MaskWarp", {
    "TOP_DESCRIPTION": "Applies various warping effects to the mask, creating distortions and movement.",
    "ADDITIONAL_INFO": """
- `Warp Type`: The type of warping effect to apply. Each creates a different distortion pattern.
  - Options: "perlin" (noise-based), "radial" (circular), "swirl" (spiral)
- `Frequency`: Controls the scale of the warping effect (0.01 to 1.0). Higher values create more rapid changes in the warp pattern.
- `Amplitude`: Controls the strength of the warping effect (0.1 to 500.0). Higher values create more extreme distortions.
- `Octaves`: For noise-based warps, adds detail at different scales (1 to 8). More octaves create more complex, detailed patterns.
"""
})

add_node_config("MIDILoadAndExtract", {
    "TOP_DESCRIPTION": """
    Loads a MIDI file and extracts specified features for mask modulation. To use this, select the notes on the piano that you want to use to control modulations. 
    Many of the different types of information in the notes can be chosen as the driving feature.""",
    "ADDITIONAL_INFO": """
- `midi_file`: Path to the MIDI file to load and analyze
- `track_selection`: Which track(s) to analyze ("all" or specific track number)
- `attribute`: MIDI attribute to extract (e.g., "Note On/Off", "Pitchbend", "Pitch", "Aftertouch")
- `frame_rate`: Frame rate of the video to sync MIDI data with
- `video_frames`: Corresponding video frames (IMAGE type)
- `chord_only`: When true, only considers full chords (BOOLEAN)
- `notes`: IGNORE THIS. Certain limitations prevent me from hiding it completely. Love, Ryan
"""
})

add_node_config("AudioFilter", {
    "TOP_DESCRIPTION": "Applies frequency filters to audio for targeted sound processing.",
    "ADDITIONAL_INFO": """
- `audio`: Input audio to be filtered (AUDIO type)
- `filters`: Frequency filters to be applied (FREQUENCY_FILTER type). These determine which frequencies are emphasized or reduced.
"""
})

add_node_config("FrequencyFilterPreset", {
    "TOP_DESCRIPTION": "Creates preset filter chains for common audio processing tasks, simplifying complex audio manipulations.",
    "ADDITIONAL_INFO": """
- `preset`: Preset to use (e.g., "isolate_kick_drum" emphasizes low frequencies, "isolate_vocals" focuses on mid-range, "remove_rumble" cuts low frequencies)

Optional inputs:
- `previous_filter`: Previous filter chain to append to, allowing for cumulative effects
"""
})

add_node_config("FrequencyFilterCustom", {
    "TOP_DESCRIPTION": "Creates custom frequency filters.",
    "ADDITIONAL_INFO": """
- `filter_type`: Type of filter ("lowpass", "highpass", "bandpass")
- `order`: Filter order (1 to 10)
- `cutoff`: Cutoff frequency (20 to 20000 Hz)

Optional inputs:
- `previous_filter`: Previous filter chain to append to
"""
})

add_node_config("AudioFeatureVisualizer", {
    "TOP_DESCRIPTION": "***WORK IN PROGESS*** Visualizes various audio features, creating visual representations of sound characteristics.",
    "ADDITIONAL_INFO": """
- `audio`: Input audio to visualize (AUDIO type)
- `video_frames`: Corresponding video frames to overlay visualizations on (IMAGE type)
- `visualization_type`: Type of visualization to generate:
  - "waveform": Shows amplitude over time
  - "spectrogram": Displays frequency content over time
  - "mfcc": Mel-frequency cepstral coefficients, useful for speech recognition
  - "chroma": Represents pitch classes, useful for harmonic analysis
  - "tonnetz": Tonal space representation
  - "spectral_centroid": Shows the "center of mass" of the spectrum over time
- `frame_rate`: Frame rate of the video for synchronization
"""
})

add_node_config("AudioSeparator", {
    "TOP_DESCRIPTION": "Separates an input audio track into its component parts using the Open-Unmix model.",
    "ADDITIONAL_INFO": """
- `audio`: Input audio to be separated (AUDIO type)
- `video_frames`: Corresponding video frames (IMAGE type)
- `frame_rate`: Frame rate of the video for synchronization

Outputs:
- Original audio
- Isolated drums track
- Isolated vocals track
- Isolated bass track
- Isolated other instruments track
- FeaturePipe containing frame information

This node uses the Open-Unmix model to separate the input audio into four stems: drums, vocals, bass, and other instruments. Each separated track is returned as an individual AUDIO type output, along with the original audio and a FeaturePipe for further processing.
"""
})

add_node_config("SpringJointSetting", {
    "TOP_DESCRIPTION": "Defines the behavior of spring joints attached to particles.",
    "ADDITIONAL_INFO": """
- `stiffness`: Stiffness of the spring (0.0 to 1000.0). Higher values create stronger connections.
- `damping`: Damping factor of the spring (0.0 to 100.0). Higher values create more resistance to motion.
- `rest_length`: Rest length of the spring (0.0 to 100.0). Longer springs allow for more stretching.
- `max_distance`: Maximum distance the spring can stretch (0.0 to 500.0). Larger values allow for more elasticity.
"""
})

add_node_config("StaticBody", {
    "TOP_DESCRIPTION": "Defines static bodies in the simulation that particles can interact with (think walls, barrier, ramps, etc.).",
    "ADDITIONAL_INFO": """
- `shape_type`: Type of shape ("segment" or "polygon")
- `x1`, `y1`, `x2`, `y2`: Coordinates defining the shape
- `elasticity`: Bounciness of the static body (0.0 to 1.0). Higher values create more bouncy collisions.
- `friction`: Friction of the static body (0.0 to 1.0). Higher values create more resistance to motion.
- `draw`: Whether to visualize the static body and how thick
- `color`: Color of the static body (RGB tuple)
"""
})

add_node_config("GravityWell", {
    "TOP_DESCRIPTION": "An optional input for a simulation space. These can be chained together to add many to a simulation.",
    "ADDITIONAL_INFO": """
- `x`: X-coordinate of the gravity well (0.0 to 1.0)
- `y`: Y-coordinate of the gravity well (0.0 to 1.0)
- `strength`: Strength of the gravity well. Higher values create stronger attraction or repulsion.
- `radius`: Radius of effect for the gravity well. Larger values affect a wider area.
- `type`: Type of gravity well ('attract' or 'repel'). Attract pulls particles in, repel pushes them away.
- `color`: Color of the gravity well visualization (RGB tuple)
- `draw`: Thickness of the gravity well visualization (0.0 to 1.0)
"""
})

add_node_config("Vortex", {
    "TOP_DESCRIPTION": "An optional input for a simulation space. These can be chained together to add many to a simulation",
    "ADDITIONAL_INFO": """
- `x`: X-coordinate of the vortex center (0.0 to 1.0)
- `y`: Y-coordinate of the vortex center (0.0 to 1.0)
- `strength`: Strength of the vortex effect (0.0 to 1000.0). Higher values create stronger swirling motion.
- `radius`: Radius of effect for the vortex (10.0 to 500.0). Larger values create wider swirling areas.
- `inward_factor`: Factor controlling how quickly particles are pulled towards the center (0.0 to 1.0). Higher values create tighter spirals.
- `movement_speed`: Speed of movement of the vortex object (0.0 to 10.0). Higher values make the vortex move faster in the simulation space.
- `color`: Color of the vortex visualization (RGB tuple)
- `draw`: Thickness of the vortex visualization (0.0 to 1.0)
"""
})

add_node_config("ParticleModulationBase", {
    "TOP_DESCRIPTION": "Base class for particle modulation settings.",
    "ADDITIONAL_INFO": """
- `start_frame`: Frame to start the modulation effect (0 to 1000)
- `end_frame`: Frame to end the modulation effect (0 to 1000)
- `effect_duration`: Duration of the modulation effect in frames (0 to 1000)
- `temporal_easing`: Easing function for the modulation effect ("ease_in_out", "linear", "bounce", "elastic", "none")
- `palindrome`: Whether to reverse the modulation effect after completion (True/False)
- `random`: Selects a random value between 0 and the chosen target value and applies it per particle
- `feature`: Optionally, pass in a feature (like audio or motion) to drive the modulation of the particles
"""
})

add_node_config("ParticleSizeModulation", {
    "TOP_DESCRIPTION": "Modulates particle size over time.",
    "ADDITIONAL_INFO": """
- `target_size`: Target size for particles at the end of the modulation (0.0 to 400.0)
"""
})

add_node_config("ParticleSpeedModulation", {
    "TOP_DESCRIPTION": "Modulates particle speed over time.",
    "ADDITIONAL_INFO": """
- `target_speed`: Target speed for particles at the end of the modulation (0.0 to 1000.0)
"""
})

add_node_config("ParticleColorModulation", {
    "TOP_DESCRIPTION": "Modulates particle color over time.",
    "ADDITIONAL_INFO": """
- `target_color`: Target color for particles at the end of the modulation (RGB tuple)
"""
})

add_node_config("FlexMaskBase", {
    "BASE_DESCRIPTION": """
- `feature`: The feature used to modulate the mask operation (FEATURE type)
- `feature_pipe`: The feature pipe containing frame information (FEATURE_PIPE type)
- `feature_threshold`: Threshold for feature activation (0.0 to 1.0)

"""
})

add_node_config("FlexMaskDepthChamber", {
    "TOP_DESCRIPTION": "Applies a depth-based mask modulation using a depth map and specified front and back depth values.",
    "ADDITIONAL_INFO": """
- `depth_map`: Input depth map (IMAGE type)
- `z_front`: Front depth value for the mask (0.0 to 1.0). Default is 1.0.
- `z_back`: Back depth value for the mask (0.0 to 1.0). Default is 0.0.
- `feature_param`: Parameter to modulate based on the feature. Options are "none", "z_front", "z_back", "both".
- `feature_mode`: Mode of feature modulation.

This node creates a mask based on the depth values in the input depth map. The mask is modulated by the specified front and back depth values, and can be further adjusted using a feature input to dynamically change the depth range.
"""
})


add_node_config("FlexMaskMorph", {
    "TOP_DESCRIPTION": "Applies morphological operations to the mask, modulated by a selected feature.",
    "ADDITIONAL_INFO": """
- `morph_type`: The type of morphological operation to apply.
  - Options: "erode", "dilate", "open", "close"
- `max_kernel_size`: Maximum size of the morphological kernel (3 to 21, odd numbers only). Larger values create more pronounced effects.
- `max_iterations`: Maximum number of times to apply the operation (1 to 50). More iterations create more extreme effects.

The strength of the morphological operation is determined by the selected feature's value at each frame.
"""
})

add_node_config("FlexMaskWarp", {
    "TOP_DESCRIPTION": "Applies warping effects to the mask, modulated by a selected feature.",
    "ADDITIONAL_INFO": """
- `warp_type`: The type of warping effect to apply. Each creates a different distortion pattern.
  - Options: "perlin" (noise-based), "radial" (circular), "swirl" (spiral)
- `frequency`: Controls the scale of the warping effect (0.01 to 1.0). Higher values create more rapid changes in the warp pattern.
- `max_amplitude`: Maximum amplitude of the warping effect (0.1 to 500.0). Higher values create more extreme distortions.
- `octaves`: For noise-based warps, adds detail at different scales (1 to 8). More octaves create more complex, detailed patterns.

The intensity of the warping effect is determined by the selected feature's value at each frame.
"""
})

add_node_config("FlexMaskTransform", {
    "TOP_DESCRIPTION": "Applies geometric transformations to the mask, modulated by a selected feature.",
    "ADDITIONAL_INFO": """
- `transform_type`: The type of transformation to apply.
  - Options: "translate", "rotate", "scale"
- `max_x_value`: Maximum horizontal component of the transformation (-1000.0 to 1000.0). Positive values move right, negative left.
- `max_y_value`: Maximum vertical component of the transformation (-1000.0 to 1000.0). Positive values move up, negative down.

The extent of the transformation is determined by the selected feature's value at each frame.
"""
})

add_node_config("FlexMaskMath", {
    "TOP_DESCRIPTION": "Performs mathematical operations between two masks, modulated by a selected feature.",
    "ADDITIONAL_INFO": """
- `mask_b`: Second mask to combine with the input mask.
- `combination_method`: Mathematical operation to apply.
  - Options: "add", "subtract", "multiply", "minimum", "maximum"

The strength of the combination is determined by the selected feature's value at each frame.
"""
})

add_node_config("FlexMaskOpacity", {
    "TOP_DESCRIPTION": "Applies opacity modulation to the mask based on a selected feature.",
    "ADDITIONAL_INFO": """
- `max_opacity`: Maximum opacity to apply to the mask (0.0 to 1.0). Higher values allow for more opaque masks.

The actual opacity applied is determined by the product of max_opacity, feature value, and strength.

This node is useful for creating masks that fade in and out based on the selected feature, allowing for smooth transitions and effects that respond to various inputs like audio, time, or other extracted features.
"""
})

add_node_config("FlexMaskVoronoiScheduled", {
    "TOP_DESCRIPTION": "Generates a Voronoi noise mask with parameters modulated by a selected feature according to a specified formula.",
    "ADDITIONAL_INFO": """
- `distance_metric`: Method used to calculate distances in the Voronoi diagram. Options include various mathematical norms and custom patterns.
- `scale`: Base scale of the Voronoi cells (0.1 to 10.0). Larger values create bigger cells.
- `detail`: Number of Voronoi cells (10 to 1000). More cells create more intricate patterns.
- `randomness`: Degree of randomness in cell placement (0.0 to 5.0). Higher values create more chaotic patterns.
- `seed`: Random seed for reproducible results (0 to 2^64 - 1).
- `x_offset`: Horizontal offset of the Voronoi pattern (-1000.0 to 1000.0).
- `y_offset`: Vertical offset of the Voronoi pattern (-1000.0 to 1000.0).
- `feature_param`: Which parameter to modulate based on the feature ("scale", "detail", "randomness", "seed", "x_offset", "y_offset").
- `formula`: Mathematical formula used to map the feature value to the feature parameter. Options include "Linear", "Quadratic", "Cubic", "Sinusoidal", and "Exponential".
- `a` and `b`: Parameters for fine-tuning the chosen formula (0.1 to 10.0).

Credit for the  heavy lifting for this node goes to https://github.com/alanhuang67/
"""
})

add_node_config("FlexMaskBinary", {
    "TOP_DESCRIPTION": "Applies binary thresholding to the mask, modulated by a selected feature.",
    "ADDITIONAL_INFO": """
- `threshold`: Base threshold value for binarization (0.0 to 1.0). Pixels above this value become white, below become black.
- `method`: Thresholding method to use.
  - Options: "simple" (basic threshold), "adaptive" (local adaptive threshold), "hysteresis" (double threshold with connectivity), "edge" (Canny edge detection)
- `max_smoothing`: Maximum amount of Gaussian smoothing to apply (0 to 51, odd values only). Higher values create smoother masks.
- `max_edge_enhancement`: Maximum strength of edge enhancement (0.0 to 10.0). Higher values create more pronounced edges.
- `feature_param`: Which parameter to modulate based on the feature value.
  - Options: "threshold" (adjusts threshold), "smoothing" (adjusts smoothing), "edge_enhancement" (adjusts edge enhancement), "none" (no modulation)

The binary mask operation is applied with strength determined by the selected feature's value at each frame. This node is useful for creating sharp, high-contrast masks that can be dynamically adjusted based on various inputs like audio, time, or other extracted features.
"""
})

add_node_config("FlexMaskEmanatingRings", {
    "TOP_DESCRIPTION": "Creates dynamic, expanding ring patterns that emanate from the edges of the input mask.",
    "ADDITIONAL_INFO": """
- `num_rings`: Number of concentric rings to generate (1 to 50). More rings create a denser, more complex pattern.
- `max_ring_width`: Maximum width of each ring as a fraction of the total distance (0.01 to 0.9). Larger values create wider, more prominent rings that overlap more.
- `wave_speed`: Speed at which the rings expand outward (0.01 to 0.5). Higher values create faster-moving, more dynamic patterns.
- `feature_param`: Determines which aspect of the effect is modulated by the input feature.
  - Options: "num_rings" (varies ring count), "ring_width" (adjusts ring thickness), "wave_speed" (changes expansion rate), "all" (modulates all parameters)

This node creates a mesmerizing effect of rings expanding from the edges of the input mask. The rings start thick at the mask boundary and thin out as they move outward, creating a pulsating, wave-like appearance. The effect can be subtle and smooth or bold and dynamic depending on the parameter settings.

The feature input can be used to dynamically adjust the effect over time, allowing for rhythmic pulsing (e.g., synced to audio) or gradual evolution of the pattern. When the feature value is below the threshold, the animation continues but no new rings are generated, creating a smooth transition effect.
"""
})

add_node_config("FlexMaskRandomShapes", {
    "TOP_DESCRIPTION": "Generates dynamic masks with randomly placed shapes, modulated by a selected feature.",
    "ADDITIONAL_INFO": """
- `max_num_shapes`: Maximum number of shapes to generate (1 to 100). The actual number is modulated by the feature if 'num_shapes' is selected as the feature parameter.
- `max_shape_size`: Maximum size of shapes as a fraction of the frame size (0.01 to 1.0). The actual size is modulated by the feature if 'shape_size' is selected as the feature parameter.
- `appearance_duration`: Number of frames over which shapes appear (1 to 100). Modulated by the feature if selected as the feature parameter.
- `disappearance_duration`: Number of frames over which shapes disappear (1 to 100). Modulated by the feature if selected as the feature parameter.
- `appearance_method`: How shapes appear and disappear.
  - Options: "grow" (shapes grow/shrink), "pop" (shapes appear/disappear suddenly), "fade" (shapes fade in/out)
- `easing_function`: Determines the rate of change for appearance/disappearance.
  - Options: "linear", "ease_in_out", "bounce", "elastic"
- `shape_type`: Type of shape to generate. Includes various geometric shapes and a "random" option.
- `feature_param`: Aspect of the effect modulated by the input feature.
  - Options: "num_shapes", "shape_size", "appearance_duration", "disappearance_duration"

This node creates a dynamic mask with randomly placed shapes that appear and disappear over time. The number, size, and timing of the shapes can be modulated by the input feature, creating effects that respond to various inputs like audio, time, or other extracted features. The shapes can grow, pop, or fade in and out, with different easing functions for smooth or bouncy transitions.
"""
})

add_node_config("FlexMaskWavePropagation", {
    "TOP_DESCRIPTION": "Good luck with this one...Simulates wave-like abstract distortions propagating from the edges of the input mask.",
    "ADDITIONAL_INFO": """
- `wave_speed`: Controls the rate of wave propagation (0.1 to 100.0). Higher values create faster-moving, more rapidly evolving patterns.
- `wave_amplitude`: Determines the intensity of the wave effect (0.1 to 2.0). Larger values create more pronounced, exaggerated distortions.
- `wave_decay`: Rate at which waves fade out over time (0.9 to 10.0). Lower values cause waves to dissipate quickly, while higher values allow waves to persist and interact more.
- `wave_frequency`: Frequency of the wave oscillations (0.01 to 10.0). Higher values create more rapid, ripple-like effects, while lower values produce smoother, more gradual undulations.
- `max_wave_field`: Maximum allowed intensity for the wave field (10.0 to 10000.0). This parameter prevents the effect from becoming too extreme over time.

This node creates a dynamic, fluid-like effect where waves seem to emanate from the edges of the input mask. The waves propagate outward, interacting with each other and creating complex, evolving patterns. The effect can range from subtle, water-like ripples to intense, psychedelic distortions depending on the parameter settings.

The wave propagation is particularly sensitive to the interplay between `wave_speed`, `wave_amplitude`, and `wave_decay`. High speed with low decay can create a turbulent, chaotic effect, while lower speed with higher decay produces a more serene, flowing appearance.

The feature input modulates the intensity of new waves being generated, allowing for dynamic control over the effect's strength. This can be used to create pulsating effects synchronized with audio or other time-varying inputs.
"""
})

add_node_config("FeatureExtractorBase", {
    "BASE_DESCRIPTION": """
 Features are used to modulate other RyanOnTheInside nodes. 

 You can replace this feature with any of the others, and it will work.
 
 Available features include Audio, Motion, MIDI, Pitch, Proximity, Depth, Time, Color, Brightness, and more.

### Parameters:
- `frame_rate`: Frame rate of the video
- `frame_count`: Total number of frames
"""
})

add_node_config("TimeFeatureNode", {
    "TOP_DESCRIPTION": "Produces a feature that changes over time based on the selected effect type. This can be used to create dynamic, time-varying mask modulations.",
    "ADDITIONAL_INFO": """
- `effect_type`: Type of time-based pattern to apply.
  - Options: "smooth" (gradual), "accelerate" (speeds up), "pulse" (rhythmic), "sawtooth" (repeating ramp), "bounce" (up and down)
- `speed`: How quickly the effect progresses (0.1 to 10.0, default: 1.0). Higher values create faster changes.
- `offset`: Shifts the starting point of the effect (0.0 to 1.0, default: 0.0). Useful for staggering multiple effects.


"""
})

add_node_config("AudioFeatureNode", {
    "TOP_DESCRIPTION": "Analyzes the input audio to extract the specified feature. This feature can then be used to modulate masks based on audio characteristics.",
    "ADDITIONAL_INFO": """
- `audio`: Input audio to analyze (AUDIO type)
- `feature_type`: Type of audio feature to extract.
  - Options: "amplitude_envelope", "rms_energy", "spectral_centroid", "onset_detection", "chroma_features"

"""
})

add_node_config("DepthFeatureNode", {
    "TOP_DESCRIPTION": "Analyzes the input depth maps to extract the specified depth-related feature. This feature can be used to modulate masks based on depth information in the scene.",
    "ADDITIONAL_INFO": """
- `depth_maps`: Input depth maps to analyze (IMAGE type)
- `feature_type`: Type of depth feature to extract.
  - Options: "mean_depth", "depth_variance", "depth_range", "gradient_magnitude", "foreground_ratio", "midground_ratio", "background_ratio"

"""
})

add_node_config("ColorFeatureNode", {
    "TOP_DESCRIPTION": "Extracts color-related features from video frames for mask modulation.",
    "ADDITIONAL_INFO": """
- `feature_type`: Type of color feature to extract
  - Options: "dominant_color" (most prevalent color), "color_variance" (variation in colors), "saturation" (color intensity), "red_ratio" (proportion of red), "green_ratio" (proportion of green), "blue_ratio" (proportion of blue)

Analyzes the input video frames to extract the specified color-related feature. This feature can be used to modulate masks based on color information in the scene, creating effects that respond to color changes over time.
"""
})

add_node_config("BrightnessFeatureNode", {
    "TOP_DESCRIPTION": "Extracts brightness-related features from video frames for mask modulation.",
    "ADDITIONAL_INFO": """
- `feature_type`: Type of brightness feature to extract
  - Options: "mean_brightness" (average brightness), "brightness_variance" (variation in brightness), "dark_ratio" (proportion of dark areas), "mid_ratio" (proportion of mid-tone areas), "bright_ratio" (proportion of bright areas)

Analyzes the input video frames to extract the specified brightness-related feature. This feature can be used to modulate masks based on lighting changes in the scene, allowing for effects that respond to overall brightness or specific tonal ranges.
"""
})

add_node_config("MotionFeatureNode", {
    "TOP_DESCRIPTION": "Extracts motion-related features from video frames for mask modulation.",
    "ADDITIONAL_INFO": """
- `feature_type`: Type of motion feature to extract
  - Options: "mean_motion" (average motion), "max_motion" (peak motion), "motion_direction" (overall direction), "horizontal_motion" (left-right movement), "vertical_motion" (up-down movement), "motion_complexity" (intricacy of motion)
- `flow_method`: Technique used for optical flow calculation
  - Options: "Farneback" (dense flow), "LucasKanade" (sparse flow), "PyramidalLK" (multi-scale sparse flow)
- `flow_threshold`: Minimum motion magnitude to consider (0.0 to 10.0). Higher values ignore subtle movements.
- `magnitude_threshold`: Relative threshold for motion magnitude (0.0 to 1.0). Higher values focus on areas of stronger motion.

Analyzes the input video frames to extract the specified motion-related feature using optical flow techniques. This feature can be used to modulate masks based on movement in the scene, creating effects that respond to motion intensity, direction, or complexity.
"""
})

add_node_config("AudioFeatureExtractor", {
    "TOP_DESCRIPTION": "Analyzes the input audio to extract the specified feature. The resulting feature can be used to modulate masks based on audio characteristics.",
    "ADDITIONAL_INFO": """
- `audio`: Input audio to analyze (AUDIO type)
- `feature_pipe`: Feature pipe for frame information (FEATURE_PIPE type)
- `feature_type`: Type of audio feature to extract
  - Options: "amplitude_envelope", "rms_energy", "spectral_centroid", "onset_detection", "chroma_features"

"""
})

add_node_config("PitchFeatureExtractor", {
    "TOP_DESCRIPTION": "Extracts pitch-related features from audio input.",
    "ADDITIONAL_INFO": """
- `audio`: Input audio to analyze (AUDIO type)
- `feature_pipe`: Feature pipe containing frame information (FEATURE_PIPE type)
- `feature_type`: Type of pitch feature to extract:
  - "pitch_filtered": Filtered pitch values
  - "pitch_direction": Direction of pitch changes
  - "vibrato_signal": Vibrato signal
  - "vibrato_intensity": Intensity of vibrato
- `window_size`: Size of the analysis window (0 for default)
- `pitch_tolerance`: Tolerance for pitch detection (0.0 to 1.0)
- `pitch_range_collections`: (Optional) Collections of pitch ranges to consider (PITCH_RANGE_COLLECTION type)

This node extracts various pitch-related features from the input audio, which can be used for further analysis or mask modulation.
"""
})

add_node_config("PitchRangeByNoteNode", {
    "TOP_DESCRIPTION": "Creates pitch ranges based on specified MIDI notes.",
    "ADDITIONAL_INFO": """
- `chord_only`: If true, only detects when all specified notes are present simultaneously (BOOLEAN)
- `pitch_tolerance_percent`: Tolerance percentage for pitch detection (0.0 to 100.0)
- `notes`: IGNORE THIS. Certain limitations prevent me from hiding it completely. Love, Ryan
- `previous_range_collection`: (Optional) Previous pitch range collection to append to (PITCH_RANGE_COLLECTION type)

This node creates pitch ranges based on specified MIDI notes, which can be used for targeted pitch detection in audio analysis.
"""
})

add_node_config("PitchRangePresetNode", {
    "TOP_DESCRIPTION": "Creates preset pitch ranges for common vocal ranges.",
    "ADDITIONAL_INFO": """
- `preset`: Preset vocal range to use:
  - Options: "Bass", "Baritone", "Tenor", "Alto", "Mezzo-soprano", "Soprano", "Contralto"
- `previous_range_collection`: (Optional) Previous pitch range collection to append to (PITCH_RANGE_COLLECTION type)

This node provides preset pitch ranges corresponding to common vocal ranges, which can be used for voice-specific audio analysis.
"""
})

add_node_config("PitchRangeNode", {
    "TOP_DESCRIPTION": "Creates a custom pitch range for audio analysis.",
    "ADDITIONAL_INFO": """
- `min_pitch`: Minimum frequency of the pitch range in Hz (20.0 to 2000.0)
- `max_pitch`: Maximum frequency of the pitch range in Hz (20.0 to 2000.0)
- `previous_range_collection`: (Optional) Previous pitch range collection to append to (PITCH_RANGE_COLLECTION type)

This node allows you to create a custom pitch range by specifying the minimum and maximum frequencies. This can be useful for targeting specific frequency ranges in audio analysis, such as isolating particular instruments or vocal ranges.

The created pitch range can be combined with other pitch ranges or used independently in pitch-related feature extraction nodes.
"""
})

add_node_config("EmptyImageFromAudio", {
    "TOP_DESCRIPTION": "Creates an empty image sequence based on audio input.",
    "ADDITIONAL_INFO": """
- `audio`: Input audio (AUDIO type)
- `frame_rate`: Frame rate of the output image sequence (0.1 to 120 fps)
- `height`: Height of the output images (16 to 4096 pixels)
- `width`: Width of the output images (16 to 4096 pixels)

This node generates an empty image sequence dimensions and frame rate, based on the duration of the input audio. 
It's useful for creating a blank canvas for further image processing or visualization that matches the length of an audio track.
"""
})

add_node_config("EmptyMaskFromAudio", {
    "TOP_DESCRIPTION": "Creates an empty mask sequence based on audio input.",
    "ADDITIONAL_INFO": """
- `audio`: Input audio (AUDIO type)
- `frame_rate`: Frame rate of the output mask sequence (0.1 to 120 fps)
- `height`: Height of the output masks (16 to 4096 pixels)
- `width`: Width of the output masks (16 to 4096 pixels)

This node generates an empty mask sequence with the specified dimensions and frame rate, based on the duration of the input audio. 
It's useful for creating a blank mask for further processing or effects that match the length of an audio track.
"""
})

add_node_config("TimeFeatureNode", {
    "TOP_DESCRIPTION": "Generates time-based features for mask modulation.",
    "ADDITIONAL_INFO": """
- `video_frames`: Input video frames (IMAGE type)
- `frame_rate`: Frame rate of the video
- `effect_type`: Type of time-based pattern to apply
  - Options: "smooth" (gradual), "accelerate" (speeds up), "pulse" (rhythmic), "sawtooth" (repeating ramp), "bounce" (up and down)
- `speed`: How quickly the effect progresses (0.1 to 10.0, default: 1.0). Higher values create faster changes.
- `offset`: Shifts the starting point of the effect (0.0 to 1.0, default: 0.0). Useful for staggering multiple effects.

Generates a feature that changes over time based on the selected effect type. This can be used to create dynamic, time-varying mask modulations.
"""
})

add_node_config("DepthFeatureNode", {
    "TOP_DESCRIPTION": "Extracts depth-related features from depth maps for mask modulation.",
    "ADDITIONAL_INFO": """
- `video_frames`: Input video frames (IMAGE type)
- `frame_rate`: Frame rate of the video
- `depth_maps`: Input depth maps to analyze (IMAGE type)
- `feature_type`: Type of depth feature to extract
  - Options: "mean_depth", "depth_variance", "depth_range", "gradient_magnitude", "foreground_ratio", "midground_ratio", "background_ratio"

Analyzes the input depth maps to extract the specified depth-related feature. This feature can be used to modulate masks based on depth information in the scene.
"""
})

add_node_config("AreaFeatureNode", {
    "TOP_DESCRIPTION": "Extracts area-related features from mask sequences for mask modulation.",
    "ADDITIONAL_INFO": """
- `masks`: Input mask sequence to analyze (MASK type)
- `feature_type`: Type of area feature to extract
  - Options: 
    - "total_area" (sum of pixels above threshold)
    - "largest_contour" (area of the largest contiguous region)
    - "bounding_box" (area of the bounding box containing the largest region)
- `threshold`: Threshold value for considering pixels as part of the area (0.0 to 1.0)

This node analyzes the input mask sequence to extract the specified area-related feature. The resulting feature can be used to modulate masks based on changes in area over time, allowing for effects that respond to the size or extent of masked regions in the scene.
"""
})

add_node_config("ProximityFeatureNode", {
    "TOP_DESCRIPTION": "Calculates a proximity feature based on the distance between anchor and query locations in video frames.",
    "ADDITIONAL_INFO": """
- `video_frames`: Input video frames (IMAGE type)
- `frame_rate`: Frame rate of the video (FLOAT type, default: 30.0, min: 1.0, max: 120.0, step: 0.1)
- `anchor_locations`: Locations of anchor points (LOCATION type)
- `query_locations`: Locations of query points (LOCATION type)
- `distance_metric`: Distance metric to use for calculation (Options: "euclidean", "manhattan", "chebyshev")

This node calculates a proximity feature based on the specified distance metric between anchor and query locations in the input video frames. The resulting feature can be used to modulate other effects based on spatial relationships.
"""
})

add_node_config("LocationFromMask", {
    "TOP_DESCRIPTION": "This is for use with proximity features. This generates locations from mask inputs using various methods.",
    "ADDITIONAL_INFO": """
- `masks`: Input masks (MASK type)
- `method`: Method to use for location extraction (Options: "mask_center", "mask_boundary", "mask_top_left", "mask_bottom_right")
- `depth_maps`: (Optional) Input depth maps. The depth map provides a value for the z coordinate of every location. If no depth map is provided, the value defaults to .5. The z coordinate is far less granular than x and y, as all we have are relative normalized depth per frame (0 to 1).  

This node generates locations from the input masks using the specified method. The locations can be used as anchor or query points for proximity calculations or other spatially dependent effects.
"""
})

add_node_config("LocationFromPoint", {
    "TOP_DESCRIPTION": "Generates locations from specified x, y, and z coordinates.",
    "ADDITIONAL_INFO": """
- `x`: X-coordinate of the location (FLOAT, default: 0.0, min: 0.0, step: 0.01)
- `y`: Y-coordinate of the location (FLOAT, default: 0.0, min: 0.0, step: 0.01)
- `batch_count`: Number of locations to generate (INT, default: 1, min: 1)
- `z`: Z-coordinate of the location (FLOAT, default: 0.0, min: 0.0, max: 1.0, step: 0.01)

This node generates a batch of locations based on the specified x, y, and z coordinates.
"""
})

add_node_config("LocationTransform", {
    "TOP_DESCRIPTION": "Transforms locations based on a feature and specified transformation type.",
    "ADDITIONAL_INFO": """
- `locations`: Input locations to be transformed (LOCATION type)
- `feature`: Feature used to modulate the transformation (FEATURE type)
- `transformation_type`: Type of transformation to apply ("translate" or "scale")
- `transformation_value`: Value of the transformation (FLOAT, default: 1.0)

This node transforms the input locations based on the specified transformation type and value, modulated by the input feature.
"""
})

add_node_config("EmitterMovement", {
    "TOP_DESCRIPTION": """These parameters work together to create complex, periodic movements for particle emitters. 
By adjusting frequencies and amplitudes, you can achieve various patterns like circles, 
figure-eights, or more chaotic motions. The direction parameters add extra dynamism by 
altering the angle of particle emission over time.""",
    "ADDITIONAL_INFO": """
Position Control:
- `emitter_x_frequency`: How quickly the emitter moves horizontally (0.0 to 10.0). Higher values create faster side-to-side motion.
- `emitter_x_amplitude`: Maximum horizontal distance the emitter moves (0.0 to 0.5). Larger values create wider movements.
- `emitter_y_frequency`: How quickly the emitter moves vertically (0.0 to 10.0). Higher values create faster up-and-down motion.
- `emitter_y_amplitude`: Maximum vertical distance the emitter moves (0.0 to 0.5). Larger values create taller movements.
Direction Control:
- `direction_frequency`: How quickly the emission angle changes (0.0 to 10.0). Higher values create more rapid direction changes.
- `direction_amplitude`: Maximum angle change in degrees (0.0 to 180.0). Larger values allow for more extreme direction shifts.

Feature Modulation:
- `feature`: Optional feature to modulate the movement (FEATURE type)
- `feature_param`: Parameter to be modulated by the feature ("emitter_x_frequency", "emitter_y_frequency", or "direction_frequency")


"""
})

add_node_config("ParticleEmitter", {
    "TOP_DESCRIPTION": "This node creates a particle emitter with the specified properties. It can be used in conjunction with particle system mask nodes to create complex particle effects. They can be chained together to add many to a given simulation.",
    "ADDITIONAL_INFO": """
- `emitter_x`: X-coordinate of the emitter (0.0 to 1.0, left to right)
- `emitter_y`: Y-coordinate of the emitter (0.0 to 1.0, up to down)
- `particle_direction`: Direction of particle emission in degrees (0.0 to 360.0, clockwise)
- `particle_spread`: Spread angle of particle emission in degrees (0.0 to 360.0, clockwise)
- `particle_size`: Size of emitted particles (1.0 to 400.0)
- `particle_speed`: Speed of emitted particles (1.0 to 1000.0)
- `emission_rate`: Rate of particle emission (0.1 to 100.0)
- `color`: Color of emitted particles (RGB string)
- `initial_plume`: Initial burst of particles (0.0 to 1.0)
- `start_frame`: Frame to start the emission (0 to 10000)
- `end_frame`: Frame to end the emission (0 to 10000)
- `emission_radius`: Defaulting to 0 (a point), this value changes the radius of the area from which the particles are emitted. The open 'mouth' of the emitter.

Optional inputs:
- `emitter_movement`: Movement settings for the emitter (EMITTER_MOVEMENT type)
- `spring_joint_setting`: Spring joint configuration for particles (SPRING_JOINT_SETTING type)
- `particle_modulation`: Modulation settings for particle properties over time (PARTICLE_MODULATION type)
"""
})

add_node_config("MovingShape", {
    "TOP_DESCRIPTION": "Generate animated mask sequences featuring a moving shape with customizable parameters.",
    "ADDITIONAL_INFO": """
- `frame_width`: Width of each frame (1-3840 pixels)
- `frame_height`: Height of each frame (1-2160 pixels)
- `num_frames`: Number of frames in the sequence (1-120)
- `rgb`: Color of the shape in RGB format, e.g., "(255,255,255)"
- `shape`: Shape type ("square", "circle", or "triangle")
- `shape_width_percent`: Width of the shape as a percentage of frame width (0-100%)
- `shape_height_percent`: Height of the shape as a percentage of frame height (0-100%)
- `shape_start_position_x`: Starting X position of the shape (-100 to 100)
- `shape_start_position_y`: Starting Y position of the shape (-100 to 100)
- `shape_end_position_x`: Ending X position of the shape (-100 to 100)
- `shape_end_position_y`: Ending Y position of the shape (-100 to 100)
- `movement_type`: Type of movement ("linear", "ease_in_out", "bounce", or "elastic")
- `grow`: Growth factor of the shape during animation (0-100)
- `palindrome`: Whether to reverse the animation sequence (True/False)
- `delay`: Number of static frames at the start (0-60)

This node creates a mask sequence with a moving shape, allowing for various animations and transformations.
"""
})

add_node_config("TextMaskNode", {
    "TOP_DESCRIPTION": "Generate mask and image sequences featuring customizable text.",
    "ADDITIONAL_INFO": """
- `width`: Width of the output image (1-8192 pixels)
- `height`: Height of the output image (1-8192 pixels)
- `text`: The text to be rendered
- `font`: Font to be used (selectable from system fonts)
- `font_size`: Size of the font (1-1000)
- `font_color`: Color of the text in RGB format, e.g., "(255,255,255)"
- `background_color`: Color of the background in RGB format, e.g., "(0,0,0)"
- `x_position`: Horizontal position of the text (0.0-1.0, where 0.5 is center)
- `y_position`: Vertical position of the text (0.0-1.0, where 0.5 is center)
- `rotation`: Rotation angle of the text (0-360 degrees)
- `max_width_ratio`: Maximum width of text as a ratio of image width (0.1-1.0)
- `batch_size`: Number of images to generate in the batch (1-10000)
"""
})

add_node_config("_mfc", {
    "TOP_DESCRIPTION": "Basic mask from color.",
    "ADDITIONAL_INFO": """
This is an abstract base class that provides common functionality for mask function components.
It is not meant to be used directly but serves as a foundation for other mask-related nodes.

Key features:
- Implements common methods for mask operations
- Provides a structure for derived classes to follow
- Ensures consistency across different mask function components
"""
})

add_node_config("DownloadOpenUnmixModel", {
    "TOP_DESCRIPTION": "Downloads and loads Open Unmix models for audio classification",
    "ADDITIONAL_INFO": """
-umxl (default) trained on private stems dataset of compressed stems. Note, that the weights are only licensed for non-commercial use (CC BY-NC-SA 4.0).

-umxhq trained on MUSDB18-HQ which comprises the same tracks as in MUSDB18 but un-compressed which yield in a full bandwidth of 22050 Hz.


"""
})

add_node_config("FeatureMixer", {
    "TOP_DESCRIPTION": "Advanced feature modulation node for fine-tuning and shaping feature values.",
    "ADDITIONAL_INFO": """
- `feature`: Input feature to be processed (FEATURE type)
- `base_gain`: Overall amplification of the feature values (0.0 to 10.0). Higher values increase the overall intensity.
- `floor`: Minimum value for the processed feature (0.0 to 1.0). Prevents values from going below this threshold.
- `ceiling`: Maximum value for the processed feature (0.0 to 10.0). Caps values at this upper limit.
- `peak_sharpness`: Sharpness of peaks in the feature curve (0.1 to 10.0). Higher values create more pronounced peaks.
- `valley_sharpness`: Sharpness of valleys in the feature curve (0.1 to 10.0). Higher values create deeper valleys.
- `attack`: Speed at which the envelope follower responds to increasing values (0.01 to 1.0). Lower values create slower attack.
- `release`: Speed at which the envelope follower responds to decreasing values (0.01 to 1.0). Lower values create slower release.
- `smoothing`: Amount of smoothing applied to the final curve (0.0 to 1.0). Higher values create smoother transitions.

This node provides extensive control over feature modulation, allowing for complex shaping of feature values over time. It combines multiple processing stages including gain, waveshaping, envelope following, and smoothing to create highly customized feature curves for mask modulation.

Outputs:
- Processed FEATURE
- Visualization of the processed feature (IMAGE type)
"""
})

add_node_config("FeatureRebase", {
    "TOP_DESCRIPTION": "Rebases feature values within specified thresholds.",
    "ADDITIONAL_INFO": """
- `feature`: Input feature to be rebased (FEATURE type)
- `lower_threshold`: Lower threshold for feature values (FLOAT, default: 0.0, min: 0.0, max: 1.0, step: 0.01)
- `upper_threshold`: Upper threshold for feature values (FLOAT, default: 1.0, min: 0.0, max: 1.0, step: 0.01)
- `invert_output`: Whether to invert the output feature values (BOOLEAN, default: False)

This node rebases the input feature values within the specified thresholds and normalizes them.
"""
})

add_node_config("FeatureMath", {
    "TOP_DESCRIPTION": "Performs mathematical operations between a feature's values and a float value.",
    "ADDITIONAL_INFO": """
- `feature`: Input feature (FEATURE type)
- `y`: Input value (FLOAT type)
- `operation`: Mathematical operation to perform ("add", "subtract", "multiply", "divide", "max", "min"). Determines how the feature's values are combined with y.

This node takes a feature and performs the specified operation between its values and the float value y, returning the processed feature and its visualization.
"""
})


add_node_config("FeatureScaler", {
    "TOP_DESCRIPTION": "Scales and transforms feature values using various mathematical functions.",
    "ADDITIONAL_INFO": """
    - `feature`: Input feature to be scaled (FEATURE type)
    - `scale_type`: Type of scaling to apply ("linear", "logarithmic", "exponential", "inverse")
    - `min_output`: Minimum output value after scaling (0.0 to 1.0)
    - `max_output`: Maximum output value after scaling (0.0 to 1.0)
    - `exponent`: Exponent for exponential scaling (0.1 to 10.0)
    """
})

add_node_config("FeatureCombine", {
    "TOP_DESCRIPTION": "Performs mathematical operations between two features.",
    "ADDITIONAL_INFO": """
    - `feature1`: First input feature (FEATURE type)
    - `feature2`: Second input feature (FEATURE type)
    - `operation`: Mathematical operation to perform ("add", "subtract", "multiply", "divide", "max", "min"). Determines how the two features are combined.
    - `weight1`: Weight applied to feature1 (0.0 to 1.0). Higher values give more importance to feature1 in the operation.
    - `weight2`: Weight applied to feature2 (0.0 to 1.0). Higher values give more importance to feature2 in the operation.
    """
})

add_node_config("FeatureSmoothing", {
    "TOP_DESCRIPTION": "Applies various smoothing techniques to a feature.",
    "ADDITIONAL_INFO": """
    - `feature`: Input feature to be smoothed (FEATURE type)
    - `smoothing_type`: Type of smoothing to apply ("moving_average", "exponential", "gaussian")
    - `window_size`: Size of the smoothing window for moving average and gaussian (3 to 21, odd numbers only). Larger values create smoother transitions but may reduce responsiveness.
    - `alpha`: Smoothing factor for exponential smoothing (0.0 to 1.0). Higher values make the feature respond more quickly to changes, while lower values create a more gradual, smoothed effect.
    - `sigma`: Standard deviation for gaussian smoothing (0.1 to 5.0). Higher values create a more pronounced smoothing effect.
    """
})

add_node_config("FeatureOscillator", {
    "TOP_DESCRIPTION": "Generates oscillating patterns based on the input feature.",
    "ADDITIONAL_INFO": """
    - `feature`: Input feature to base oscillation on (FEATURE type)
    - `oscillator_type`: Type of oscillation ("sine", "square", "sawtooth", "triangle")
    - `frequency`: Frequency of oscillation (0.1 to 10.0)
    - `amplitude`: Amplitude of oscillation (0.0 to 1.0)
    - `phase_shift`: Phase shift of oscillation (0.0 to 2π)
    - `blend`: Blend factor between original feature and oscillation (0.0 to 1.0)
    """
})

add_node_config("FeatureFade", {
    "TOP_DESCRIPTION": "Fades between two features based on a fader value or a control feature.",
    "ADDITIONAL_INFO": """
- `feature1`: First input feature to fade from (FEATURE type)
- `feature2`: Second input feature to fade to (FEATURE type)
- `fader`: Static fader value to control the blend between feature1 and feature2 (0.0 to 1.0). 0.0 is 100 percent feature1, 1.0 is 100 percent feature2.
- `control_feature`: Optional feature to dynamically control the fader value (FEATURE type). If provided, the fader value will be modulated by this feature.

Shoutout @cyncratic
"""
})

add_node_config("FeatureTruncateOrExtend", {
    "TOP_DESCRIPTION": "Adjusts the length of a feature to match a target feature pipe, either by truncating or extending it.",
    "ADDITIONAL_INFO": """
- `feature`: Input feature to be adjusted (FEATURE type)
- `target_feature_pipe`: Target feature pipe to match the length (FEATURE type)
- `fill_method`: Method to use when extending the feature:
  - "zeros": Fills with 0's
  - "ones": Fills with 1's
  - "average": Fills with the average value of the source feature
  - "random": Fills with random values between 0 and 1
  - "repeat": Repeats the source values from the beginning
- `invert_output`: Whether to invert the output feature values (True/False)

This node adjusts the length of the input feature to match the length of the target feature pipe. If the input feature is longer, it will be truncated. If it's shorter, it will be extended using the specified fill method.

The "repeat" fill method is particularly useful for maintaining patterns or rhythms when extending the feature.

Use cases:
1. Adapting audio-extracted features for shorter or longer video animations
2. Synchronizing features of different lengths for complex animations
3. Creating looping patterns by repeating shorter features

Outputs:
- Adjusted FEATURE
- Visualization of the adjusted feature (IMAGE type)
"""
})


add_node_config("FeatureToWeightsStrategy", {
    "TOP_DESCRIPTION": "Converts a FEATURE input into a WEIGHTS_STRATEGY for use with IPAdapter nodes.",
    "ADDITIONAL_INFO": """
- `feature`: Input feature to be converted (FEATURE type)

This node takes a FEATURE input and converts it into a WEIGHTS_STRATEGY that can be used with IPAdapter nodes. It creates a custom weights strategy based on the feature values for each frame.
This node is particularly useful for creating dynamic, feature-driven animations with IPAdapter, where the strength of the adaptation can vary over time based on extracted features from audio, video, or other sources.

"""
})

NODE_CONFIGS["FlexImageBase"] = {
    "BASE_DESCRIPTION": """
- `images`: Input image sequence (IMAGE type)
- `feature`: Feature used to modulate the effect (FEATURE type)
- `feature_pipe`: Feature pipe containing frame information (FEATURE_PIPE type)
- `strength`: Overall strength of the effect (0.0 to 1.0)
- `feature_threshold`: Minimum feature value to apply the effect (0.0 to 1.0)
- `feature_mode`: How the feature modulates the parameter ("relative" or "absolute"). Relative mode adjusts the parameter based on its current value, while absolute mode directly sets the parameter to the feature value.
"""
}

add_node_config("FlexImageContrast", {
    "TOP_DESCRIPTION": "Adjusts the contrast and brightness of the image, with an option to preserve luminosity.",
    "ADDITIONAL_INFO": """
- `contrast`: Controls the amount of contrast adjustment (0.0 to 3.0). Values greater than 1.0 increase contrast, while values less than 1.0 decrease it.
- `brightness`: Adjusts the overall brightness of the image (-1.0 to 1.0). Positive values brighten the image, negative values darken it.
- `preserve_luminosity`: When enabled, maintains the overall luminosity of the image after applying contrast and brightness adjustments (True/False).
- `feature_param`: Parameter to modulate based on the feature. Options are "contrast", "brightness", "preserve_luminosity", "None".

This node allows for dynamic adjustment of image contrast and brightness, with the ability to preserve the overall luminosity of the image. It's useful for enhancing image details, adjusting exposure, or creating dramatic lighting effects.

The contrast adjustment is applied around the mean value of the image, which helps maintain the overall balance of the image. The brightness adjustment is applied uniformly across the image.

When 'preserve_luminosity' is enabled, the node calculates and adjusts the final luminosity to match the original image, which can help prevent over-brightening or over-darkening when applying strong contrast adjustments.

Use cases include:
1. Enhancing low-contrast images
2. Creating high-contrast, dramatic effects
3. Correcting under or overexposed images
4. Dynamically adjusting image tone based on audio or other features
"""
})

add_node_config("FlexImageEdgeDetect", {
    "TOP_DESCRIPTION": "Applies edge detection to the image using the Canny algorithm.",
    "ADDITIONAL_INFO": """
- `low_threshold`: Lower bound for the hysteresis thresholding (0 to 255).
- `high_threshold`: Upper bound for the hysteresis thresholding (0 to 255).
- `feature_param`: Parameter to modulate based on the feature. Options are "low_threshold", "high_threshold", "None".

This node detects edges in the image using the Canny edge detection algorithm. The thresholds control the sensitivity of the edge detection.
"""
})

add_node_config("FlexImagePosterize", {
    "TOP_DESCRIPTION": "Applies a posterization effect to the image, reducing the number of colors.",
    "ADDITIONAL_INFO": """
- `max_levels`: Maximum number of color levels per channel (2 to 256).
- `dither_strength`: Intensity of dithering effect (0.0 to 1.0).
- `channel_separation`: Degree of separation between color channels (0.0 to 1.0).
- `gamma`: Gamma correction applied before posterization (0.1 to 2.2).
- `feature_param`: Parameter to modulate based on the feature. Options are "max_levels", "dither_strength", "channel_separation", "gamma", "None".

This node reduces the number of colors in the image, creating a posterized effect. Dithering can be applied to reduce banding.
"""
})

add_node_config("FlexImageKaleidoscope", {
    "TOP_DESCRIPTION": "Creates a kaleidoscope effect by mirroring and rotating segments of the image.",
    "ADDITIONAL_INFO": """
- `segments`: Number of mirror segments (2 to 32).
- `center_x`: X-coordinate of the effect center (0.0 to 1.0).
- `center_y`: Y-coordinate of the effect center (0.0 to 1.0).
- `zoom`: Zoom factor for the effect (0.1 to 2.0).
- `rotation`: Rotation angle of the effect (0.0 to 360.0 degrees).
- `precession`: Rate of rotation change over time (-1.0 to 1.0).
- `speed`: Speed of the effect animation (0.1 to 5.0).
- `feature_param`: Parameter to modulate based on the feature. Options are "segments", "zoom", "rotation", "precession", "speed", "None".

This node creates a kaleidoscope effect by mirroring and rotating segments of the image.
"""
})

add_node_config("FlexImageColorGrade", {
    "TOP_DESCRIPTION": "Applies color grading to the image using a Look-Up Table (LUT).",
    "ADDITIONAL_INFO": """
- `intensity`: Strength of the color grading effect (0.0 to 1.0).
- `mix`: Blend factor between original and graded image (0.0 to 1.0).
- `lut_file`: Path to the LUT file (optional).
- `feature_param`: Parameter to modulate based on the feature. Options are "intensity", "mix", "None".

This node applies color grading to the image using a LUT. The intensity and mix parameters control the strength and blend of the effect.
"""
})

add_node_config("FlexImageGlitch", {
    "TOP_DESCRIPTION": "Creates a glitch effect by applying horizontal shifts and color channel separation.",
    "ADDITIONAL_INFO": """
- `shift_amount`: Magnitude of horizontal shift (0.0 to 1.0).
- `scan_lines`: Number of scan lines to add (0 to 100).
- `color_shift`: Amount of color channel separation (0.0 to 1.0).
- `feature_param`: Parameter to modulate based on the feature. Options are "shift_amount", "scan_lines", "color_shift", "None".

This node creates a glitch effect by shifting pixels horizontally and separating color channels.
"""
})

add_node_config("FlexImageChromaticAberration", {
    "TOP_DESCRIPTION": "Simulates chromatic aberration by shifting color channels.",
    "ADDITIONAL_INFO": """
- `shift_amount`: Magnitude of color channel shift (0.0 to 0.1).
- `angle`: Angle of the shift effect (0.0 to 360.0 degrees).
- `feature_param`: Parameter to modulate based on the feature. Options are "shift_amount", "angle", "None".

This node simulates chromatic aberration by shifting the red and blue color channels in opposite directions.
"""
})

add_node_config("FlexImagePixelate", {
    "TOP_DESCRIPTION": "Applies a pixelation effect to the image.",
    "ADDITIONAL_INFO": """
- `pixel_size`: Size of each pixelated block (1 to 100 pixels).
- `feature_param`: Parameter to modulate based on the feature. Options are "pixel_size", "None".

This node reduces the resolution of the image by applying a pixelation effect.
"""
})

add_node_config("FlexImageBloom", {
    "TOP_DESCRIPTION": "Adds a bloom effect to bright areas of the image.",
    "ADDITIONAL_INFO": """
- `threshold`: Brightness threshold for the bloom effect (0.0 to 1.0).
- `blur_amount`: Amount of blur applied to the bloom (0.0 to 50.0).
- `intensity`: Strength of the bloom effect (0.0 to 1.0).
- `feature_param`: Parameter to modulate based on the feature. Options are "threshold", "blur_amount", "intensity", "None".

This node adds a bloom effect to bright areas of the image, creating a glowing effect.
"""
})

add_node_config("FlexImageTiltShift", {
    "TOP_DESCRIPTION": "Creates a tilt-shift effect, simulating a shallow depth of field.",
    "ADDITIONAL_INFO": """
- `blur_amount`: Strength of the blur effect (0.0 to 50.0).
- `focus_position_x`: X-coordinate of the focus center (0.0 to 1.0).
- `focus_position_y`: Y-coordinate of the focus center (0.0 to 1.0).
- `focus_width`: Width of the focus area (0.0 to 1.0).
- `focus_height`: Height of the focus area (0.0 to 1.0).
- `focus_shape`: Shape of the focus area ("rectangle" or "ellipse").
- `feature_param`: Parameter to modulate based on the feature. Options are "blur_amount", "focus_position_x", "focus_position_y", "focus_width", "focus_height", "None".

This node creates a tilt-shift effect, simulating a shallow depth of field by blurring areas outside the focus region.
"""
})

add_node_config("FlexImageParallax", {
    "TOP_DESCRIPTION": "Applies a parallax effect to the image using a depth map.",
    "ADDITIONAL_INFO": """
- `shift_x`: Horizontal shift factor for the parallax effect (-1.0 to 1.0). Positive values shift right, negative values shift left.
- `shift_y`: Vertical shift factor for the parallax effect (-1.0 to 1.0). Positive values shift up, negative values shift down.
- `depth_map`: Input depth map (IMAGE type). The depth map is used to determine the amount of shift for each pixel.
- `feature_param`: Parameter to modulate based on the feature. Options are "shift_x", "shift_y", "None".

This node creates a parallax effect by shifting pixels in the image based on the corresponding values in the depth map. The shift factors `shift_x` and `shift_y` control the direction and magnitude of the parallax effect. The depth map should be provided as an image, where the intensity of each pixel represents the depth value.
"""
})

add_node_config("UtilityNode",{
    "BASE_DESCRIPTION":"Various Utils"
})

add_node_config("ImageIntervalSelect", {
    "TOP_DESCRIPTION": "Selects images from a sequence at specified intervals.",
    "ADDITIONAL_INFO": """
- `image`: Input image sequence (IMAGE type)
- `interval`: Interval at which to select images (1 to 100000)
- `start_at`: Starting index for selection (0 to 100000)
- `end_at`: Ending index for selection (0 to 100000)
"""
})

add_node_config("ImageIntervalSelectPercentage", {
    "TOP_DESCRIPTION": "Selects images from a sequence at specified percentage intervals.",
    "ADDITIONAL_INFO": """
- `image`: Input image sequence (IMAGE type)
- `interval_percentage`: Interval at which to select images as a percentage of the total sequence length (1 to 100)
- `start_percentage`: Starting percentage for selection (0 to 100)
- `end_percentage`: Ending percentage for selection (0 to 100)
"""
})

add_node_config("DepthInjection", {
    "TOP_DESCRIPTION": "Modifies depth maps based on mask contours, creating spherical gradients.",
    "ADDITIONAL_INFO": """
- `depth_map`: Input depth map (IMAGE type)
- `mask`: Input mask to define areas for depth modification (MASK type)
- `gradient_steepness`: Controls the steepness of the spherical gradient (0.1 to 10.0). Higher values create sharper transitions.
- `depth_min`: Minimum depth value for the modified areas (0.0 to 1.0)
- `depth_max`: Maximum depth value for the modified areas (0.0 to 1.0)
- `strength`: Overall strength of the depth modification effect (0.0 to 1.0)

This node modifies depth maps by creating spherical gradients based on the contours of the input mask. It's useful for adding depth variations to specific areas of an image, such as creating a sense of volume for masked objects.

The process involves:
1. Finding contours in the mask
2. Generating spherical gradients for each contour
3. Scaling the gradients to the specified depth range
4. Blending the modified depth with the original depth map

This node can be particularly effective for:
- Adding depth to flat objects in a scene
- Creating a sense of volume for masked areas
- Fine-tuning depth maps for more realistic 3D effects

Note: The node currently doesn't use the feature modulation capabilities, but these could be added in future versions for dynamic depth modifications.
"""
})

add_node_config("ImageChunks", {
    "TOP_DESCRIPTION": "Concatenates images into a grid.",
    "ADDITIONAL_INFO": """
- `image`: Input image sequence (IMAGE type)
- `padding`: Padding between images in the grid (default: 0)
- `normalize`: Whether to normalize the images (default: False)
- `scale_each`: Whether to scale each image individually (default: False)
- `pad_value`: Value for padding (default: 0)
"""
})

add_node_config("VideoChunks", {
    "TOP_DESCRIPTION": "Chunks images into grids.",
    "ADDITIONAL_INFO": """
- `image`: Input image sequence (IMAGE type)
- `chunk_size`: Number of images per grid (default: 4, min: 1)
- `padding`: Padding between images in the grid (default: 2)
- `normalize`: Whether to normalize the images (default: False)
- `scale_each`: Whether to scale each image individually (default: False)
- `pad_value`: Value for padding (default: 0)
"""
})

add_node_config("SwapDevice", {
    "TOP_DESCRIPTION": "Transfers the image and mask tensors to the specified device (CPU or CUDA).",
    "ADDITIONAL_INFO": """
- `device`: The target device to transfer the tensors to. Options are "cpu" or "cuda".
- `image`: (Optional) The image tensor to transfer. If not provided, a zero tensor is created.
- `mask`: (Optional) The mask tensor to transfer. If not provided, a zero tensor is created.

This node checks if the specified device is available and transfers the image and mask tensors to that device. If the device is not available, it raises a ValueError.
"""
})

add_node_config("ImageShuffle", {
    "TOP_DESCRIPTION": "Shuffles images in groups of specified size.",
    "ADDITIONAL_INFO": """
- `image`: Input image sequence (IMAGE type)
- `shuffle_size`: Number of images per shuffle group (default: 4, min: 1)
"""
})

add_node_config("ImageDifference", {
    "TOP_DESCRIPTION": "Computes the difference between consecutive images.",
    "ADDITIONAL_INFO": """
- `image`: Input image sequence (IMAGE type)
"""
})

add_node_config("EffectVisualizer", {
    "TOP_DESCRIPTION": "Visualizes feature values on video frames.",
    "ADDITIONAL_INFO": """
- `video_frames`: Input video frames (IMAGE type)
- `feature`: Feature to visualize (FEATURE type)
- `text_color`: Color for text overlay (RGB string, e.g., "(255,255,255)")
- `font_scale`: Scale factor for the font size (0.1 to 2.0)
"""
})

add_node_config("ProximityVisualizer", {
    "TOP_DESCRIPTION": "Visualizes proximity relationships between anchor and query locations on video frames.",
    "ADDITIONAL_INFO": """
- `anchor_locations`: Locations of anchor points (LOCATION type)
- `query_locations`: Locations of query points (LOCATION type)
- `feature`: Proximity feature to visualize (FEATURE type)
- `anchor_color`: Color for anchor points (RGB string, e.g., "(255,0,0)")
- `query_color`: Color for query points (RGB string, e.g., "(0,255,0)")
- `line_color`: Color for the line connecting closest points (RGB string, e.g., "(0,0,255)")


The visualization helps in understanding spatial relationships and proximity-based effects in the video sequence.
"""
})

add_node_config("PitchVisualizer", {
    "TOP_DESCRIPTION": "Visualizes pitch-related information on video frames.",
    "ADDITIONAL_INFO": """
- `video_frames`: Input video frames (IMAGE type)
- `feature`: Pitch feature to visualize (FEATURE type)
- `text_color`: Color of the text overlay (RGB string, e.g., "(255,255,255)")
- `font_scale`: Scale factor for the font size (0.1 to 2.0)

This node overlays pitch-related information on video frames, including:
- Feature value
- Pitch value in Hz
- Confidence value of the pitch detection
- Approximate musical note

The information is displayed as text on each frame, allowing for easy visualization of pitch characteristics alongside the video content.
"""
})

add_node_config("FlexVideoBase", {
    "TOP_DESCRIPTION": "Base class for flexible video effect nodes.",
    "ADDITIONAL_INFO": """
- `images`: Input video frames (IMAGE type)
- `feature`: Feature used to modulate the effect (FEATURE type)
- `strength`: Overall strength of the effect (0.0 to 2.0)
- `feature_mode`: How the feature modulates the parameter ("relative" or "absolute")
- `feature_param`: Parameter to be modulated by the feature
- `feature_threshold`: Minimum feature value to apply the effect (0.0 to 1.0)
- `feature_pipe`: (Sometimes Optional) Feature pipe containing frame information (FEATURE_PIPE type)
"""
})

add_node_config("FlexVideoDirection", {
    "TOP_DESCRIPTION": "Applies a direction-based effect to video frames based on feature values.",
    "ADDITIONAL_INFO": """
- Inherits parameters from FlexVideoBase
- `feature_pipe`: (Optional) Feature pipe containing frame information (FEATURE_PIPE type)

This node creates a video effect by selecting frames based on the input feature values. It can create effects like reversing, speeding up, or creating loops in the video sequence.
"""
})

add_node_config("FlexVideoSpeed", {
    "TOP_DESCRIPTION": "Adjusts the playback speed of video frames based on feature values.",
    "ADDITIONAL_INFO": """
- Inherits parameters from FlexVideoBase
- `feature_pipe`: Feature pipe containing frame information (FEATURE_PIPE type)
- `max_speed_percent`: Maximum speed as a percentage of the original speed (1.0 to 1000.0)
- `duration_adjustment_method`: Method to adjust video duration ("Interpolate" or "Truncate/Repeat")

This node modifies the playback speed of the video based on the input feature values. It can create effects like slow motion, fast forward, or variable speed playback.
"""
})

add_node_config("FlexVideoFrameBlend", {
    "TOP_DESCRIPTION": "Applies frame blending effect to video frames based on feature values.",
    "ADDITIONAL_INFO": """
- Inherits parameters from FlexVideoBase
- `blend_strength`: Strength of the frame blending effect (0.0 to 1.0)

This node creates a frame blending effect, where each frame is blended with the next frame. The strength of the blend is modulated by the input feature values, allowing for dynamic motion blur effects.
"""
})