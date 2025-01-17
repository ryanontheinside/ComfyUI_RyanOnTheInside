from abc import ABC, abstractmethod
from comfy.utils import ProgressBar
import numpy as np
import torch
from ...tooltips import apply_tooltips
from .parameter_scheduling import ParameterScheduler

@apply_tooltips
class FlexBase(ABC):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = {
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
        
        return base_inputs


    def __init__(self):
        self.progress_bar = None
        self.parameter_scheduler = None
        self.frame_count = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None

    def initialize_scheduler(self, frame_count: int, **kwargs):
        """Initialize parameter scheduler with all numeric parameters"""
        self.frame_count = frame_count
        self.parameter_scheduler = ParameterScheduler(frame_count)
        for key, value in kwargs.items():
            if isinstance(value, (int, float, list, tuple)):
                self.parameter_scheduler.register_parameter(key, value)


    def get_feature_value(self, frame_index: int, feature=None, param_name=None):
        """Get feature value from either a provided feature or a scheduled parameter"""
        if feature is not None:
            return feature.get_value_at_frame(frame_index)
        elif self.parameter_scheduler and param_name:
            feature_seq = self.parameter_scheduler.get_as_feature(param_name)
            if feature_seq is not None:
                return feature_seq[frame_index]
        return None

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return []

    def modulate_param(self, param_name: str, param_value: float | list | tuple | np.ndarray,
                      feature_value: float, strength: float, mode: str,
                      frame_index: int = 0) -> float:
        """Modulate a parameter value based on a feature value.
        
        Args:
            param_name: Name of the parameter being modulated
            param_value: Value to modulate (can be single value or array-like)
            feature_value: Feature value to use for modulation (0-1)
            strength: Strength of the modulation (0-1)
            mode: Modulation mode ("relative" or "absolute")
            frame_index: Frame index for array-like parameters
            
        Returns:
            Modulated parameter value
        """
        # Handle array-like parameters
        if isinstance(param_value, (list, tuple, np.ndarray)):
            try:
                base_value = float(param_value[frame_index])
            except (IndexError, TypeError):
                base_value = float(param_value[0])
        else:
            base_value = float(param_value)

        # Apply modulation
        if mode == "relative":
            # Adjust parameter relative to its value and the feature
            return base_value * (1 + (feature_value - 0.5) * 2 * strength)
        else:  # absolute
            # Adjust parameter directly based on the feature
            return base_value * feature_value * strength

    @abstractmethod
    def apply_effect(self, *args, **kwargs):
        """Apply the effect with potential parameter scheduling"""
        pass

    @abstractmethod
    def apply_effect_internal(self, *args, **kwargs):
        """Internal method to be implemented by subclasses."""
        pass

    def process_parameters(self, frame_index: int = 0, feature_value: float = None, 
                          feature_param: str = None, feature_mode: str = "relative", **kwargs) -> dict:
        """Process parameters considering both scheduling and feature modulation"""
        # Initialize parameter scheduler if not already done
        if self.parameter_scheduler is None:
            # Determine frame count from the first array/list parameter we find
            frame_count = kwargs.get('frame_count', 1)
            for value in kwargs.values():
                if isinstance(value, (list, tuple, np.ndarray)):
                    frame_count = len(value)
                    break
            self.initialize_scheduler(frame_count, **kwargs)

        # Get input types to determine parameter types
        input_types = self.INPUT_TYPES()["required"]

        # Get all parameters that could be scheduled
        processed_kwargs = {}
        
        # Helper function to process schedulable parameters  (strength, feature_threshold)
        def process_schedulable_param(param_name: str, default_value: float) -> float:
            value = kwargs.get(param_name, default_value)
            if isinstance(value, (list, tuple, np.ndarray)):


                try:
                    value = float(value[frame_index])
                except (IndexError, TypeError):
                    value = float(value[0])
            else:
                value = float(value)
            processed_kwargs[param_name] = value
            return value

        # Process parameters needed for feature modulation
        strength = process_schedulable_param('strength', 1.0)
        feature_threshold = process_schedulable_param('feature_threshold', 0.0)

        # Process remaining parameters
        for param_name, value in kwargs.items():
            if param_name in ['strength', 'feature_threshold']:  # Skip already processed parameters
                continue
                
            # Pass through any non-numeric parameters
            if param_name not in input_types or input_types[param_name][0] not in ["INT", "FLOAT"]:
                processed_kwargs[param_name] = value
                continue

            try:
                # Handle different types of inputs
                if isinstance(value, (list, tuple, np.ndarray)):
                    # Convert numpy arrays to lists if needed
                    if isinstance(value, np.ndarray):
                        if value.ndim > 1:  # If multi-dimensional array
                            value = value.flatten().tolist()  # Flatten and convert to list
                        else:
                            value = value.tolist()
                    # Use frame_index to get the current value
                    try:
                        base_value = float(value[frame_index])
                    except (IndexError, TypeError):
                        base_value = float(value[0])  # Fallback to first value
                else:
                    # Single value
                    base_value = float(value)

                # Apply feature modulation if this is the target parameter
                if param_name == feature_param and feature_value is not None:
                    # Only apply modulation if feature value meets threshold
                    if feature_value >= feature_threshold:
                        processed_value = self.modulate_param(param_name, base_value, feature_value, strength, feature_mode)
                    else:
                        processed_value = base_value
                        
                    # Convert back to int if needed
                    if input_types[param_name][0] == "INT":
                        processed_kwargs[param_name] = int(processed_value)
                    else:
                        processed_kwargs[param_name] = processed_value
                else:
                    # Convert to int if needed
                    if input_types[param_name][0] == "INT":
                        processed_kwargs[param_name] = int(base_value)
                    else:
                        processed_kwargs[param_name] = base_value
            except (ValueError, TypeError):
                # If conversion fails, pass through unchanged
                processed_kwargs[param_name] = value

        
        processed_kwargs['feature_param'] = feature_param
        processed_kwargs['feature_mode'] = feature_mode

        return processed_kwargs

