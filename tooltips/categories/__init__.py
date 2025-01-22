"""
Category-specific tooltip definitions.
"""

from . import (
    audio,
    depth,
    doom,
    flex,
    images,
    latents,
    masks,
    misc,
    preprocessors,
    video
)

def register_all_tooltips():
    """Register tooltips from all categories."""
    audio.register_tooltips()
    depth.register_tooltips()
    doom.register_tooltips()
    flex.register_tooltips()
    images.register_tooltips()
    latents.register_tooltips()
    masks.register_tooltips()
    misc.register_tooltips()
    preprocessors.register_tooltips()
    video.register_tooltips()
