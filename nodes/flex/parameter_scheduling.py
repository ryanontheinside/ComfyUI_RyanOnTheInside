from typing import Union, List, Any
import numpy as np
from abc import ABC, abstractmethod
from ... import RyanOnTheInside
from ...tooltips import apply_tooltips

class ScheduledParameter:
    """Wrapper class for parameters that can be either single values or sequences"""
    def __init__(self, value: Union[float, int, List[Union[float, int]]], frame_count: int):
        self.original_value = value
        self.frame_count = frame_count
        self._sequence = None
        self._initialize_sequence()

    def _initialize_sequence(self):
        if isinstance(self.original_value, (list, tuple)):
            # If it's a sequence, interpolate to match frame count
            x = np.linspace(0, 1, len(self.original_value))
            y = np.array(self.original_value)
            f = np.interp(np.linspace(0, 1, self.frame_count), x, y)
            self._sequence = f
        else:
            # If it's a single value, repeat it
            self._sequence = np.full(self.frame_count, self.original_value)

    def get_value(self, frame_index: int) -> Union[float, int]:
        """Get the parameter value for a specific frame"""
        if frame_index < 0 or frame_index >= self.frame_count:
            raise ValueError(f"Frame index {frame_index} out of bounds [0, {self.frame_count})")
        return float(self._sequence[frame_index])

    def get_normalized_sequence(self) -> np.ndarray:
        """Get the sequence normalized to [0,1] range for feature-like behavior"""
        if self._sequence is None:
            return None
        seq = np.array(self._sequence)
        min_val = np.min(seq)
        max_val = np.max(seq)
        if max_val > min_val:
            return (seq - min_val) / (max_val - min_val)
        return np.full_like(seq, 0.5)  # If all values are the same, return 0.5

    @property
    def is_scheduled(self) -> bool:
        """Returns True if the parameter is a sequence, False if it's a single value"""
        return isinstance(self.original_value, (list, tuple))

class ParameterScheduler:
    """Helper class to manage scheduled parameters for a node"""
    def __init__(self, frame_count: int):
        self.frame_count = frame_count
        self.parameters = {}

    def register_parameter(self, name: str, value: Any) -> None:
        """Register a parameter that might be scheduled"""
        if isinstance(value, (int, float)) or (isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value)):
            self.parameters[name] = ScheduledParameter(value, self.frame_count)

    def get_value(self, name: str, frame_index: int) -> Any:
        """Get the value of a parameter for a specific frame"""
        if name in self.parameters:
            return self.parameters[name].get_value(frame_index)
        raise KeyError(f"Parameter {name} not registered")

    def get_as_feature(self, name: str) -> np.ndarray:
        """Get a parameter's sequence normalized as a feature (0-1 range)"""
        if name in self.parameters:
            return self.parameters[name].get_normalized_sequence()
        return None

    def is_scheduled(self, name: str) -> bool:
        """Check if a parameter is scheduled"""
        if name in self.parameters:
            return self.parameters[name].is_scheduled
        return False

    def has_scheduled_parameters(self) -> bool:
        """Check if any parameters are scheduled"""
        return any(param.is_scheduled for param in self.parameters.values()) 
    
    
#TODO: abstract normalize function from here and FeatureRenormalize and place in utils or something.
@apply_tooltips
class SchedulerNode(RyanOnTheInside):
    """Base class for nodes that convert features to schedulable parameters"""
    CATEGORY = "RyanOnTheInside/FlexFeatures/Scheduling"
    FUNCTION = "convert"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "invert_output": ("BOOLEAN", {"default": False}),
            }
        }
    
    def process_values(self, feature, lower_threshold, upper_threshold, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Calculate the range for normalization
        range_size = upper_threshold - lower_threshold
        
        # Use feature.min_value and feature.max_value if available, otherwise use actual min/max
        min_val = getattr(feature, 'min_value', min(values))
        max_val = getattr(feature, 'max_value', max(values))
        
        # Normalize values to fit between lower and upper threshold
        if max_val == min_val:
            normalized = [lower_threshold for _ in values]  # All values are the same
        else:
            normalized = [
                lower_threshold + (range_size * (v - min_val) / (max_val - min_val))
                for v in values
            ]
        
        if invert_output:
            normalized = [upper_threshold - (v - lower_threshold) for v in normalized]
                
        return normalized

@apply_tooltips
class FeatureToFlexIntParam(SchedulerNode):
    """Converts a feature to a schedulable integer parameter"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "lower_threshold": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "upper_threshold": ("INT", {"default": 100, "min": -10000, "max": 10000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("PARAMETER",)
    
    def convert(self, feature, lower_threshold, upper_threshold, invert_output):
        values = self.process_values(feature, lower_threshold, upper_threshold, invert_output)
        # Round to integers
        int_values = [int(round(v)) for v in values]
        return (int_values,)

@apply_tooltips
class FeatureToFlexFloatParam(SchedulerNode):
    """Converts a feature to a schedulable float parameter"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "lower_threshold": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "upper_threshold": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("PARAMETER",)
    
    def convert(self, feature, lower_threshold, upper_threshold, invert_output):
        values = self.process_values(feature, lower_threshold, upper_threshold, invert_output)
        return (values,) 