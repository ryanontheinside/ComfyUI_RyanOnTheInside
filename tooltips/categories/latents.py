"""Tooltips for latents-related nodes."""

from ..tooltip_manager import TooltipManager

def register_tooltips():
    """Register tooltips for latents nodes"""

    # FlexLatentBase tooltips (inherits from: RyanOnTheInside, FlexBase)
    TooltipManager.register_tooltips("FlexLatentBase", {
        "latents": "Input latent tensor to be processed (LATENT type)",
        "feature": "Feature used to modulate the effect (FEATURE type)",
        "feature_pipe": "Feature pipe containing frame information (FEATURE_PIPE type)",
        "feature_threshold": "Threshold for feature activation (0.0 to 1.0)"
    }, inherits_from=['RyanOnTheInside', 'FlexBase'])

    # FlexLatentInterpolate tooltips (inherits from: FlexLatentBase)
    TooltipManager.register_tooltips("FlexLatentInterpolate", {
        "latent_2": "Second latent tensor to interpolate with (LATENT type)",
        "interpolation_mode": "Method of interpolation ('Linear' or 'Spherical')"
    }, inherits_from='FlexLatentBase')

    # EmbeddingGuidedLatentInterpolate tooltips (inherits from: FlexLatentBase)
    TooltipManager.register_tooltips("EmbeddingGuidedLatentInterpolate", {
        "latent_2": "Second latent tensor to interpolate with (LATENT type)",
        "embedding_1": "First embedding tensor for guidance (EMBEDS type)",
        "embedding_2": "Second embedding tensor for guidance (EMBEDS type)",
        "interpolation_mode": "Method of interpolation ('Linear' or 'Spherical')"
    }, inherits_from='FlexLatentBase')

    # FlexLatentBlend tooltips (inherits from: FlexLatentBase)
    TooltipManager.register_tooltips("FlexLatentBlend", {
        "latent_2": "Second latent tensor to blend with (LATENT type)",
        "blend_mode": "Type of blending operation ('Add', 'Multiply', 'Screen', 'Overlay')",
        "blend_strength": "Strength of the blending effect (0.0 to 1.0)"
    }, inherits_from='FlexLatentBase')

    # FlexLatentNoise tooltips (inherits from: FlexLatentBase)
    TooltipManager.register_tooltips("FlexLatentNoise", {
        "noise_level": "Amount of noise to add (0.0 to 1.0)",
        "noise_type": "Type of noise to generate ('Gaussian' or 'Uniform')"
    }, inherits_from='FlexLatentBase')

    # LatentFrequencyBlender tooltips (inherits from: FlexLatentBase)
    TooltipManager.register_tooltips("LatentFrequencyBlender", {
        "images": "Input images to be encoded into latents (IMAGE type)",
        "vae": "VAE model for encoding images (VAE type)",
        "frequency_ranges": "Frequency ranges to analyze (FREQUENCY_RANGE type, multi-select)",
        "audio": "Audio input for frequency analysis (AUDIO type)",
        "feature_type": "Type of audio feature to extract ('amplitude_envelope', 'rms_energy', 'spectral_flux', 'zero_crossing_rate')",
        "strength": "Overall strength of the blending effect (0.0 to 10.0)",
        "feature_mode": "How features affect the blending ('relative' or 'absolute')",
        "frame_rate": "Frame rate for audio analysis (1.0 to 120.0 fps)",
        "nonlinear_transform": "Transform applied to feature values ('none', 'square', 'sqrt', 'log', 'exp')",
        "blending_mode": "Method for blending latents ('linear', 'slerp', 'hard_switch')"
    }, inherits_from='FlexLatentBase')
