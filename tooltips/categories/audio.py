"""Tooltips for audio-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for audio nodes"""

    # BaseAudioProcessor tooltips
    TooltipManager.register_tooltips("BaseAudioProcessor", {
        "frame_rate": "Frame rate of the video",
        "frame_count": "Total number of frames"
    })

    # FlexAudioVisualizerBase tooltips (inherits from: FlexBase, RyanOnTheInside)
    TooltipManager.register_tooltips("FlexAudioVisualizerBase", {
        "audio": "Input audio to visualize",
        "frame_rate": "Frame rate of the output visualization (1.0 to 240.0 fps)",
        "screen_width": "Width of the output visualization (100 to 1920 pixels)",
        "screen_height": "Height of the output visualization (100 to 1080 pixels)",
        "strength": "Strength of parameter modulation (0.0 to 1.0)",
        "feature_param": "Parameter to modulate based on the optional feature input",
        "feature_mode": "Mode of parameter modulation ('relative' or 'absolute')",
        "position_x": "Horizontal position of the visualization center (0.0 to 1.0)",
        "position_y": "Vertical position of the visualization center (0.0 to 1.0)",
        "opt_feature": "Optional feature input for parameter modulation"
    }, inherits_from=['FlexBase', 'RyanOnTheInside'])

    # FlexAudioVisualizerLine tooltips (inherits from: FlexAudioVisualizerBase)
    TooltipManager.register_tooltips("FlexAudioVisualizerLine", {
        "visualization_method": "Visualization style ('bar' or 'line')",
        "visualization_feature": "Data source for visualization ('frequency' or 'waveform')",
        "smoothing": "Amount of smoothing applied to the visualization (0.0 to 1.0)",
        "rotation": "Rotation angle in degrees (0.0 to 360.0)",
        "length": "Length of the visualization in pixels. 0 means auto-fit to screen edges based on rotation (0.0 to 4000.0)",
        "num_bars": "Number of bars/points in visualization (1 to 1024)",
        "max_height": "Maximum height of visualization (10.0 to 2000.0)",
        "min_height": "Minimum height of visualization (0.0 to 500.0)",
        "separation": "Separation between bars (0.0 to 100.0)",
        "curvature": "Curvature of bar corners (0.0 to 50.0)",
        "reflect": "Whether to draw visualization upward (false) or downward (true) from baseline",
        "curve_smoothing": "Smoothing for line visualization (0.0 to 1.0)",
        "fft_size": "FFT window size for frequency analysis (256 to 8192)",
        "min_frequency": "Minimum frequency to visualize (20.0 to 20000.0 Hz)",
        "max_frequency": "Maximum frequency to visualize (20.0 to 20000.0 Hz)"
    }, inherits_from='FlexAudioVisualizerBase')

    # FlexAudioVisualizerCircular tooltips (inherits from: FlexAudioVisualizerBase)
    TooltipManager.register_tooltips("FlexAudioVisualizerCircular", {
        "visualization_method": "Visualization style ('bar' or 'line')",
        "visualization_feature": "Data source for visualization ('frequency' or 'waveform')",
        "smoothing": "Amount of smoothing applied to the visualization (0.0 to 1.0)",
        "rotation": "Rotation angle in degrees (0.0 to 360.0)",
        "num_points": "Number of points in circular visualization (3 to 1000)",
        "fft_size": "FFT window size for frequency analysis (256 to 8192)",
        "min_frequency": "Minimum frequency to visualize (20.0 to 20000.0 Hz)",
        "max_frequency": "Maximum frequency to visualize (20.0 to 20000.0 Hz)",
        "radius": "Radius of visualization (10.0 to 1000.0 pixels)",
        "line_width": "Width of visualization lines (1 to 10 pixels)",
        "amplitude_scale": "Scaling factor for amplitude (1.0 to 1000.0)",
        "base_radius": "Base radius for visualization (10.0 to 1000.0 pixels)"
    }, inherits_from='FlexAudioVisualizerBase')

    # FlexAudioVisualizerContour tooltips (inherits from: FlexAudioVisualizerBase)
    TooltipManager.register_tooltips("FlexAudioVisualizerContour", {
        "visualization_method": """Visualization style:
- bar: Individual bars extending from the contour
- line: Continuous line following the contour""",
        "visualization_feature": """Data source for visualization:
- frequency: Shows frequency spectrum analysis
- waveform: Shows direct audio amplitude""",
        "smoothing": "Amount of smoothing applied to the visualization (0.0 to 1.0)",
        "rotation": "Rotation angle in degrees (0.0 to 360.0)",
        "num_points": "Number of points in contour visualization (3 to 1000)",
        "fft_size": "FFT window size for frequency analysis (256 to 8192)",
        "min_frequency": "Minimum frequency to visualize (20.0 to 20000.0 Hz)",
        "max_frequency": "Maximum frequency to visualize (20.0 to 20000.0 Hz)",
        "bar_length": "Length of bars extending from contour (1.0 to 100.0 pixels)",
        "line_width": "Width of visualization lines (1 to 10 pixels)",
        "contour_smoothing": "Amount of smoothing applied to the contour (0 to 50)",
        "mask": "Input mask to find contours from - can contain multiple distinct areas",
        "direction": """Direction of the visualization relative to the contour:
- outward: Extends away from the contour
- inward: Extends towards the center of the contour
- both: Shows both inward and outward effects simultaneously""",
        "min_contour_area": "Minimum area threshold for detecting contours (0.0 to 10000.0)",
        "max_contours": "Maximum number of contours to process (1 to 20)",
        "distribute_by": """How to distribute audio data among multiple contours:
- area: Larger contours get more data points
- perimeter: Longer contours get more data points
- equal: All contours get equal data points"""
    }, inherits_from='FlexAudioVisualizerBase', description="""Visualize audio features along mask contours with customizable effects.
    
Perfect for creating audio-reactive animations that follow the edges of masks.
Supports multiple contours and various visualization styles.""")

    # FlexAudioBase tooltips (inherits from: FlexBase, RyanOnTheInside)
    TooltipManager.register_tooltips("FlexAudioBase", {
        "audio": "Input audio to be processed",
        "target_fps": "Target frames per second for processing (1.0 to 60.0 fps)",
        "opt_feature": "Optional feature input for parameter modulation",
        "opt_feature_pipe": "Optional feature pipe for frame synchronization",
        "strength": "Overall strength of the effect (0.0 to 1.0)",
        "feature_threshold": "Minimum feature value to apply the effect (0.0 to 1.0)",
        "feature_param": "Parameter to be modulated by the feature",
        "feature_mode": "How the feature modulates the parameter ('relative' or 'absolute')"
    }, inherits_from=['FlexBase', 'RyanOnTheInside'])

    # FlexAudioPitchShift tooltips (inherits from: FlexAudioBase)
    TooltipManager.register_tooltips("FlexAudioPitchShift", {
        "n_steps": "Amount of pitch shift in semitones (0.0 to 12.0)"
    }, inherits_from='FlexAudioBase')

    # FlexAudioTimeStretch tooltips (inherits from: FlexAudioBase)
    TooltipManager.register_tooltips("FlexAudioTimeStretch", {
        "rate": "Time stretching factor (0.5 to 2.0)"
    }, inherits_from='FlexAudioBase', description="Time stretch the audio")

    #AudioNodeBase tooltips
    TooltipManager.register_tooltips("AudioNodeBase", {
        "audio": "Input audio to be processed"
    }, inherits_from='RyanOnTheInside', description="Manipulate audio in various ways")

    # AudioUtility tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("AudioUtility", {
        "audio": "Input audio to be processed"
    }, inherits_from='AudioNodeBase')

    # AudioPad tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioPad", {
        "audio": "Input audio to be processed",
        "pad_left": "Number of samples to pad on the left side (0 to 44100)",
        "pad_right": "Number of samples to pad on the right side (0 to 44100)",
        "pad_mode": "Method of padding ('constant', 'reflect', 'replicate', or 'circular')"
    }, inherits_from='AudioUtility')

    # AudioVolumeNormalization tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioVolumeNormalization", {
        "audio": "Input audio to be processed",
        "target_level": "Desired RMS level in decibels (-60.0 to 0.0 dB)"
    }, inherits_from='AudioUtility')

    # AudioResample tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioResample", {
        "audio": "Input audio to be processed",
        "new_sample_rate": "Desired new sample rate (8000 to 192000 Hz)"
    }, inherits_from='AudioUtility')

    # AudioChannelMerge tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioChannelMerge", {
        "audio_list": "List of audio channels to merge"
    }, inherits_from='AudioUtility')

    # AudioChannelSplit tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioChannelSplit", {
        "audio": "Input stereo audio to be split"
    }, inherits_from='AudioUtility')

    # AudioConcatenate tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioConcatenate", {
        "audio1": "First input audio",
        "audio2": "Second input audio"
    }, inherits_from='AudioUtility')

    # Audio_Combine tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("Audio_Combine", {
        "audio1": "First input audio",
        "audio2": "Second input audio",
        "weight1": "Weight for the first audio input (0.0 to 1.0)",
        "weight2": "Weight for the second audio input (0.0 to 1.0)"
    }, inherits_from='AudioUtility')

    # AudioSubtract tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioSubtract", {
        "audio1": "First input audio",
        "audio2": "Second input audio"
    }, inherits_from='AudioUtility')

    # AudioInfo tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioInfo", {
        "audio": "Input audio to analyze",
        "frame_rate": "Target frame rate for animation timing calculations (frames per second)",
        "total_frames": "Total number of frames based on audio duration and frame rate",
        "frames_per_beat": "Number of frames per musical beat at the detected tempo",
        "frames_per_bar": "Number of frames per musical bar (assumes 4/4 time signature)",
        "frames_per_quarter": "Number of frames per quarter note",
        "frames_per_eighth": "Number of frames per eighth note",
        "audio_duration": "Total duration of the audio in seconds",
        "beats_per_second": "Number of beats per second based on detected tempo",
        "detected_bpm": "Detected tempo in beats per minute (BPM)",
        "sample_rate": "Audio sample rate in Hz",
        "num_channels": "Number of audio channels (1 for mono, 2 for stereo)",
        "num_samples": "Total number of audio samples",
        "max_amplitude": "Maximum absolute amplitude in the audio (0.0 to 1.0)",
        "mean_amplitude": "Mean absolute amplitude across all samples (0.0 to 1.0)",
        "rms_amplitude": "Root mean square amplitude - indicates overall perceived loudness (0.0 to 1.0)",
        "bit_depth": "Bit depth of the audio data (e.g., float32, int16)"
    }, inherits_from='AudioUtility', description="""Analyzes audio to provide timing and technical information.
    
Perfect for:
- Synchronizing animations with music using detected BPM
- Planning keyframes based on musical timing
- Understanding audio properties for processing
- Creating amplitude-based effects""")

    # AudioDither tooltips (inherits from: AudioUtility)
    TooltipManager.register_tooltips("AudioDither", {
        "audio": "Input audio to apply dithering",
        "bit_depth": "Target bit depth (8 to 32)",
        "noise_shaping": "Type of noise shaping to apply ('none' or 'triangular')"
    }, inherits_from='AudioUtility')

    # AudioEffect tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("AudioEffect", {
        "audio": "Input audio to apply effect"
    }, inherits_from='AudioNodeBase')

    # AudioPitchShift tooltips (inherits from: AudioEffect)
    TooltipManager.register_tooltips("AudioPitchShift", {
        "audio": "Input audio to be processed",
        "n_steps": "Amount of pitch shift in semitones (-12 to 12)"
    }, inherits_from='AudioEffect')

    # AudioFade tooltips (inherits from: AudioEffect)
    TooltipManager.register_tooltips("AudioFade", {
        "audio": "Input audio to be processed",
        "fade_in_duration": "Duration of fade in (seconds)",
        "fade_out_duration": "Duration of fade out (seconds)",
        "shape": "Shape of the fade curve ('linear', 'exponential', 'logarithmic', 'quarter_sine', 'half_sine')"
    }, inherits_from='AudioEffect')

    # AudioGain tooltips (inherits from: AudioEffect)
    TooltipManager.register_tooltips("AudioGain", {
        "audio": "Input audio to be processed",
        "gain_db": "Gain factor in decibels (-20.0 to 20.0 dB)"
    }, inherits_from='AudioEffect')

    # AudioTimeStretch tooltips (inherits from: AudioEffect)
    TooltipManager.register_tooltips("AudioTimeStretch", {
        "audio": "Input audio to be processed",
        "rate": "Time stretching factor (0.5 to 2.0)"
    }, inherits_from='AudioEffect')

    # AudioNodeBase tooltips (inherits from: RyanOnTheInside)
    TooltipManager.register_tooltips("AudioNodeBase", {
        "audio": "Input audio to be processed"
    }, inherits_from='RyanOnTheInside')

    # DownloadOpenUnmixModel tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("DownloadOpenUnmixModel", {
        "model_type": "Model type to download ('umxl' for compressed stems, 'umxhq' for uncompressed MUSDB18-HQ)"
    }, inherits_from='AudioNodeBase')

    # AudioSeparatorSimple tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("AudioSeparatorSimple", {
        "model": "OpenUnmix model for audio separation",
        "audio": "Input audio to be separated into stems"
    }, inherits_from='AudioNodeBase')

    # AudioFilter tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("AudioFilter", {
        "audio": "Input audio to be filtered",
        "filters": "Frequency filters to be applied"
    }, inherits_from='AudioNodeBase')

    # FrequencyFilterPreset tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("FrequencyFilterPreset", {
        "preset": "Predefined filter preset to use",
        "previous_filter": "Previous filter chain to append to (optional)"
    }, inherits_from='AudioNodeBase')

    # FrequencyFilterCustom tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("FrequencyFilterCustom", {
        "filter_type": "Type of filter (lowpass, highpass, bandpass)",
        "order": "Filter order (1 to 10)",
        "cutoff": "Cutoff frequency (20 to 20000 Hz)",
        "previous_filter": "Previous filter chain to append to (optional)"
    }, inherits_from='AudioNodeBase')

    # FrequencyRange tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("FrequencyRange", {
        "low_cutoff": "Lower cutoff frequency (20 to 20000 Hz)",
        "high_cutoff": "Upper cutoff frequency (20 to 20000 Hz)",
        "order": "Filter order (1 to 10)",
        "previous_range": "Previous frequency range to append to (optional)"
    }, inherits_from='AudioNodeBase')

    # AudioFeatureVisualizer tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("AudioFeatureVisualizer", {
        "audio": "Input audio to visualize",
        "video_frames": "Video frames to overlay visualization on",
        "visualization_type": "Type of visualization (waveform, spectrogram, mfcc, chroma, tonnetz, spectral_centroid)",
        "frame_rate": "Frame rate of the visualization (0.1 to 120 fps)",
        "x_axis": "X-axis scale type",
        "y_axis": "Y-axis scale type",
        "cmap": "Color map for the visualization",
        "visualizer": "Visualization backend to use (pygame or matplotlib)"
    }, inherits_from='AudioNodeBase')

    # EmptyImageFromAudio tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("EmptyImageFromAudio", {
        "audio": "Input audio to determine frame count",
        "frame_rate": "Frame rate of the output image sequence (0.1 to 120 fps)",
        "height": "Height of the output images (16 to 4096 pixels)",
        "width": "Width of the output images (16 to 4096 pixels)"
    }, inherits_from='AudioNodeBase')

    # EmptyMaskFromAudio tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("EmptyMaskFromAudio", {
        "audio": "Input audio to determine frame count",
        "frame_rate": "Frame rate of the output mask sequence (0.1 to 120 fps)",
        "height": "Height of the output masks (16 to 4096 pixels)",
        "width": "Width of the output masks (16 to 4096 pixels)"
    }, inherits_from='AudioNodeBase')

    # EmptyImageAndMaskFromAudio tooltips (inherits from: AudioNodeBase)
    TooltipManager.register_tooltips("EmptyImageAndMaskFromAudio", {
        "audio": "Input audio to determine frame count",
        "frame_rate": "Frame rate of the output sequences (0.1 to 120 fps)",
        "height": "Height of the output images and masks (16 to 4096 pixels)",
        "width": "Width of the output images and masks (16 to 4096 pixels)"
    }, inherits_from='AudioNodeBase')


