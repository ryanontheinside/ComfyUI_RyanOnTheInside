from abc import ABC, abstractmethod
from comfy.utils import ProgressBar
import numpy as np
import torch
from ...tooltips import apply_tooltips

@apply_tooltips
class FlexBase(ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_param": (
                    cls.get_modifiable_params(), 
                    {"default": cls.get_modifiable_params()[0] if cls.get_modifiable_params() else "None"}
                ),
                "feature_mode": (["relative", "absolute"], {"default": "relative"}),
            },
            "optional": {
                "opt_feature": ("FEATURE",),
            }
        }

    CATEGORY = "RyanOnTheInside/FlexBase"
    RETURN_TYPES = ()  # To be defined by subclasses
    FUNCTION = "apply_effect"

    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return []

    def modulate_param(self, param_name, param_value, feature_value, strength, mode):
        if mode == "relative":
            # Adjust parameter relative to its value and the feature
            return param_value * (1 + (feature_value - 0.5) * 2 * strength)
        else:  # absolute
            # Adjust parameter directly based on the feature
            return param_value * feature_value * strength

    @abstractmethod
    def apply_effect(self, *args, **kwargs):
        """Main method to apply the effect."""
        pass

    @abstractmethod
    def apply_effect_internal(self, *args, **kwargs):
        """Internal method to be implemented by subclasses."""
        pass
