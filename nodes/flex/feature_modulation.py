from ... import RyanOnTheInside
import numpy as np
import torch
import random
from ..node_utilities import apply_easing
from ...tooltips import apply_tooltips
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

@apply_tooltips
class FeatureModulationBase(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexFeatures/FeatureModulators"
    FUNCTION = "modulate"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "invert_output": ("BOOLEAN", {"default": False}),
            }
        }
    
    def create_processed_feature(self, original_feature, processed_values, name_prefix="Processed", invert_output=False):
        class ProcessedFeature(type(original_feature)):
            def __init__(self, original_feature, processed_values, invert_output):
                self.__dict__.update(original_feature.__dict__)
                self.name = f"{name_prefix}_{original_feature.name}"
                if invert_output:
                    self.name = f"Inverted_{self.name}"
                self.frame_rate = original_feature.frame_rate
                self.frame_count = len(processed_values)
                
                if invert_output:
                    min_val, max_val = min(processed_values), max(processed_values)
                    self.data = [max_val - v + min_val for v in processed_values]
                else:
                    self.data = processed_values

            def extract(self):
                return self

            def get_value_at_frame(self, frame_index):
                return self.data[frame_index]

        return ProcessedFeature(original_feature, processed_values, invert_output)

@apply_tooltips
class FeatureMixer(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "base_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ceiling": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "peak_sharpness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "valley_sharpness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "attack": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "release": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rise_detection_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rise_smoothing_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.05}),
                **super().INPUT_TYPES()["required"],  # Include the invert_output option
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "modulate"

    def modulate(self, feature, base_gain, floor, ceiling, peak_sharpness, valley_sharpness, attack, release, smoothing, feature_threshold, rise_detection_threshold, rise_smoothing_factor, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        values = [v if v >= feature_threshold else 0 for v in values]
        
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            normalized = [0 for _ in values]  # All values are the same, normalize to 0
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        
        gained = [v * base_gain for v in normalized]
        
        def waveshape(v):
            return v**peak_sharpness if v > 0.5 else 1 - (1-v)**valley_sharpness
        waveshaped = [waveshape(v) for v in gained]
        
        def apply_envelope(values, attack, release):
            envelope = []
            current = values[0]
            for v in values:
                if v > current:
                    current += (v - current) * attack
                else:
                    current += (v - current) * release
                envelope.append(current)
            return envelope
        enveloped = apply_envelope(waveshaped, attack, release)
        
        def smooth_values(values, smoothing_factor):
            smoothed = values.copy()
            for i in range(1, len(values)):
                smoothed[i] = smoothed[i-1] + (values[i] - smoothed[i-1]) * (1 - smoothing_factor)
            return smoothed
        smoothed = smooth_values(enveloped, smoothing)
        
        # try to do look ahead
        adjusted = self.apply_rise_time_adjustment(smoothed, rise_detection_threshold, rise_smoothing_factor)
        
        # chop
        final_values = [max(floor, min(ceiling, v)) for v in adjusted]
        
        processed_feature = self.create_processed_feature(feature, final_values, "Processed", invert_output)
        return (processed_feature,)

    def apply_rise_time_adjustment(self, values, rise_detection_threshold, rise_smoothing_factor):
        if all(v == 0 for v in values):
            return values
        
        adjusted = values.copy()
        window_size = 5

        for i in range(len(values) - window_size):
            if values[i + window_size] - values[i] > rise_detection_threshold:
                
                start_index = i
                end_index = min(len(values), i + 2*window_size)
                peak_index = start_index + np.argmax(values[start_index:end_index])
                peak_value = values[peak_index]

               
                for j in range(start_index, peak_index):
                    progress = (j - start_index) / (peak_index - start_index)
                    smoothed_value = values[start_index] + (peak_value - values[start_index]) * (progress ** (1/rise_smoothing_factor))
                    adjusted[j] = max(adjusted[j], smoothed_value)  # Take the max to prevent lowering existing higher values

        return adjusted
    
@apply_tooltips
class FeatureScaler(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "scale_type": (["linear", "logarithmic", "exponential", "inverse"],),
                "min_output": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_output": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exponent": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)

    def modulate(self, feature, scale_type, min_output, max_output, exponent, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        min_val, max_val = min(values), max(values)
        normalized = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
        
        if scale_type == "linear":
            scaled = normalized
        elif scale_type == "logarithmic":
            scaled = [np.log1p(v) / np.log1p(1) for v in normalized]
        elif scale_type == "exponential":
            scaled = [v ** exponent for v in normalized]
        elif scale_type == "inverse":
            scaled = [1 - v for v in normalized]
        
        final_values = [min_output + v * (max_output - min_output) for v in scaled]
        
        processed_feature = self.create_processed_feature(feature, final_values, "Scaled", invert_output)
        return (processed_feature,)

@apply_tooltips
class FeatureCombine(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature1": ("FEATURE",),
                "feature2": ("FEATURE",),
                "operation": (["add", "subtract", "multiply", "divide", "max", "min"],),
                "weight1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)

    def modulate(self, feature1, feature2, operation, weight1, weight2, invert_output):
        values1 = [feature1.get_value_at_frame(i) for i in range(feature1.frame_count)]
        values2 = [feature2.get_value_at_frame(i) for i in range(feature2.frame_count)]
        
        # Ensure both features have the same length
        min_length = min(len(values1), len(values2))
        values1 = values1[:min_length]
        values2 = values2[:min_length]
        
        if operation == "add":
            combined = [weight1 * v1 + weight2 * v2 for v1, v2 in zip(values1, values2)]
        elif operation == "subtract":
            combined = [weight1 * v1 - weight2 * v2 for v1, v2 in zip(values1, values2)]
        elif operation == "multiply":
            combined = [weight1 * v1 * weight2 * v2 for v1, v2 in zip(values1, values2)]
        elif operation == "divide":
            combined = [weight1 * v1 / (weight2 * v2) if v2 != 0 else 0 for v1, v2 in zip(values1, values2)]
        elif operation == "max":
            combined = [max(weight1 * v1, weight2 * v2) for v1, v2 in zip(values1, values2)]
        elif operation == "min":
            combined = [min(weight1 * v1, weight2 * v2) for v1, v2 in zip(values1, values2)]
        
        processed_feature = self.create_processed_feature(feature1, combined, "Combined", invert_output)
        return (processed_feature,)

@apply_tooltips
class FeatureMath(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000000.0, "step": 0.005}),
                "operation": (["add", "subtract", "multiply", "divide", "max", "min"],),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "modulate"

    def modulate(self, feature, y, operation, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        if operation == "add":
            result = [v + y for v in values]
        elif operation == "subtract":
            result = [v - y for v in values]
        elif operation == "multiply":
            result = [v * y for v in values]
        elif operation == "divide":
            result = [v / y if y != 0 else 0 for v in values]
        elif operation == "max":
            result = [max(v, y) for v in values]
        elif operation == "min":
            result = [min(v, y) for v in values]
        
        processed_feature = self.create_processed_feature(feature, result, "MathResult", invert_output)
        return (processed_feature,)
    
@apply_tooltips
class FeatureSmoothing(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "smoothing_type": (["moving_average", "exponential", "gaussian"],),
                "window_size": ("INT", {"default": 5, "min": 3, "max": 21, "step": 2}),
                "alpha": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)

    def modulate(self, feature, smoothing_type, window_size, alpha, sigma, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        original_min = min(values)
        
        if smoothing_type == "moving_average":
            smoothed = np.convolve(values, np.ones(window_size), 'valid') / window_size
            # Pad the beginning and end to maintain the original length
            pad = (len(values) - len(smoothed)) // 2
            smoothed = np.pad(smoothed, (pad, pad), mode='edge')
        elif smoothing_type == "exponential":
            smoothed = [values[0]]
            for value in values[1:]:
                smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
        elif smoothing_type == "gaussian":
            x = np.arange(-window_size // 2 + 1, window_size // 2 + 1)
            kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
            kernel = kernel / np.sum(kernel)
            smoothed = np.convolve(values, kernel, mode='same')
        
        # Adjust the smoothed values to ensure the minimum value remains unchanged
        smoothed_min = min(smoothed)
        adjustment = original_min - smoothed_min
        adjusted_smoothed = [v + adjustment for v in smoothed]
        
        processed_feature = self.create_processed_feature(feature, adjusted_smoothed, "Smoothed", invert_output)
        return (processed_feature,)

@apply_tooltips
class FeatureOscillator(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "oscillator_type": (["sine", "square", "sawtooth", "triangle"],),
                "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "phase_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2*np.pi, "step": 0.1}),
                "blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)

    def modulate(self, feature, oscillator_type, frequency, amplitude, phase_shift, blend, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        t = np.linspace(0, 2*np.pi, len(values))
        
        if oscillator_type == "sine":
            oscillation = amplitude * np.sin(frequency * t + phase_shift)
        elif oscillator_type == "square":
            oscillation = amplitude * np.sign(np.sin(frequency * t + phase_shift))
        elif oscillator_type == "sawtooth":
            oscillation = amplitude * ((t + phase_shift) % (2*np.pi) / np.pi - 1)
        elif oscillator_type == "triangle":
            oscillation = amplitude * (2 / np.pi * np.arcsin(np.sin(frequency * t + phase_shift)))
        
        blended = [v * (1 - blend) + osc * blend for v, osc in zip(values, oscillation)]
        
        processed_feature = self.create_processed_feature(feature, blended, "Oscillated", invert_output)
        return (processed_feature,)


#NOTE  separated from FeatureMath for ease  of  use.
@apply_tooltips
class FeatureFade(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature1": ("FEATURE",),
                "feature2": ("FEATURE",),
                "fader": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                **super().INPUT_TYPES()["required"],
            },
            "optional": {
                "control_feature": ("FEATURE",),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)

    def modulate(self, feature1, feature2, fader, invert_output, control_feature=None):
        values1 = [feature1.get_value_at_frame(i) for i in range(feature1.frame_count)]
        values2 = [feature2.get_value_at_frame(i) for i in range(feature2.frame_count)]
        
        # Ensure both features have the same length
        min_length = min(len(values1), len(values2))
        values1 = values1[:min_length]
        values2 = values2[:min_length]
        
        if control_feature:
            control_values = [control_feature.get_value_at_frame(i) for i in range(control_feature.frame_count)]
            control_values = control_values[:min_length]
            fader_values = [(v - min(control_values)) / (max(control_values) - min(control_values)) if max(control_values) > min(control_values) else 0.5 for v in control_values]
        else:
            fader_values = [fader] * min_length
        
        combined = [(1 - f) * v1 + f * v2 for v1, v2, f in zip(values1, values2, fader_values)]
        
        processed_feature = self.create_processed_feature(feature1, combined, "Faded", invert_output)
        return (processed_feature,)

#NOTE: this class is technically redundant to FeatureMixer, but it's kept for clarity and ease of use.
@apply_tooltips
class FeatureRebase(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "lower_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upper_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "rebase"

    def rebase(self, feature, lower_threshold, upper_threshold, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        rebased_values = [v if lower_threshold <= v <= upper_threshold else 0 for v in values]
        
        min_val, max_val = min(rebased_values), max(rebased_values)
        if min_val == max_val:
            normalized = [0 for _ in rebased_values]  # All values are the same, normalize to 0
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in rebased_values]
        
        processed_feature = self.create_processed_feature(feature, normalized, "Rebased", invert_output)
        return (processed_feature,)

@apply_tooltips
class FeatureRenormalize(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "lower_threshold": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "upper_threshold": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                **super().INPUT_TYPES()["required"],
            }
        }
    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "renormalize"

    def renormalize(self, feature, lower_threshold, upper_threshold, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        range_size = upper_threshold - lower_threshold
        
        if max(values) == min(values):
            normalized = [lower_threshold for _ in values]
        else:
            normalized = [
                lower_threshold + (range_size * (v - min(values)) / (max(values) - min(values)))
                for v in values
            ]
        
        processed_feature = self.create_processed_feature(feature, normalized, "Renormalized", invert_output)
        return (processed_feature,)

@apply_tooltips
class FeatureTruncateOrExtend(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "target_feature_pipe": ("FEATURE_PIPE",),
                "fill_method": (["zeros", "ones", "average", "random", "repeat"],),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "truncate_or_extend"

    def truncate_or_extend(self, feature, target_feature_pipe, fill_method, invert_output):
        source_values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        target_length = target_feature_pipe.frame_count

        if len(source_values) > target_length:
            # Truncate
            adjusted_values = source_values[:target_length]
        elif len(source_values) < target_length:
            # Extend
            adjusted_values = source_values.copy()
            extension_length = target_length - len(source_values)
            
            if fill_method == "zeros":
                adjusted_values.extend([0] * extension_length)
            elif fill_method == "ones":
                adjusted_values.extend([1] * extension_length)
            elif fill_method == "average":
                avg_value = np.mean(source_values)
                adjusted_values.extend([avg_value] * extension_length)
            elif fill_method == "random":
                adjusted_values.extend([random.random() for _ in range(extension_length)])
            elif fill_method == "repeat":
                while len(adjusted_values) < target_length:
                    adjusted_values.extend(source_values[:min(len(source_values), target_length - len(adjusted_values))])
        else:
            # Same length, no adjustment needed
            adjusted_values = source_values

        processed_feature = self.create_processed_feature(feature, adjusted_values, "TruncatedOrExtended", invert_output)
        return (processed_feature,)
    

@apply_tooltips
class FeatureAccumulate(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "skip_thresholded": ("BOOLEAN", {"default": False}),
                **super().INPUT_TYPES()["required"],
                "frames_window": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "deccumulate": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "accumulate"

    def accumulate(self, feature, start, end, threshold, skip_thresholded, frames_window, deccumulate, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        accumulated = []
        
        if frames_window == 0:
            frames_window = feature.frame_count

        # Process values in windows
        for i in range(0, len(values), frames_window):
            window_values = values[i:i + frames_window]
            window_accumulated = []
            current_sum = 0
            
            # Determine direction for this window
            reverse = deccumulate and (i // frames_window) % 2 == 1
            
            if reverse:
                window_values = window_values[::-1]
            
            # Accumulate within window
            for v in window_values:
                if v >= threshold:
                    current_sum += v
                    window_accumulated.append(current_sum)
                else:
                    if skip_thresholded:
                        window_accumulated.append(v)
                    else:
                        window_accumulated.append(current_sum)
            
            if reverse:
                window_accumulated = window_accumulated[::-1]
            
            accumulated.extend(window_accumulated)
        
        # Normalize accumulated values between start and end
        min_val, max_val = min(accumulated), max(accumulated)
        if min_val == max_val:
            normalized = [start for _ in accumulated]
        else:
            normalized = [start + (v - min_val) * (end - start) / (max_val - min_val) for v in accumulated]
        
        processed_feature = self.create_processed_feature(feature, normalized, "Accumulated", invert_output)
        return (processed_feature,)
    


import numpy as np
from scipy.interpolate import interp1d

@apply_tooltips
class FeatureContiguousInterpolate(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "easing": (["linear", "ease_in_quad", "ease_out_quad", "ease_in_out_quad", 
                            "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic",
                            "ease_in_quart", "ease_out_quart", "ease_in_out_quart"],),
                "fade_out": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "interpolate"

    def interpolate(self, feature, threshold, start, end, easing, fade_out, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Identify contiguous segments
        segments = []
        current_segment = []
        for i, v in enumerate(values):
            if v >= threshold:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
        if current_segment:
            segments.append(current_segment)

        # Apply interpolation to segments and add fade-out
        interpolated = values.copy()
        for segment in segments:
            segment_length = len(segment)
            
            # Calculate the interpolation for the segment
            t = np.linspace(0, 1, segment_length)
            interpolated_values = apply_easing(t, start, end, easing)
            
            # Apply the interpolation to the segment
            for i, idx in enumerate(segment):
                interpolated[idx] = interpolated_values[i]
            
            # Apply fade-out after the segment
            if fade_out > 0:
                fade_out_start = segment[-1] + 1
                fade_out_end = min(fade_out_start + fade_out, len(values))
                fade_out_length = fade_out_end - fade_out_start
                
                if fade_out_length > 0:
                    t_fade = np.linspace(0, 1, fade_out_length)
                    fade_out_values = apply_easing(t_fade, end, start, easing)
                    
                    for i, idx in enumerate(range(fade_out_start, fade_out_end)):
                        interpolated[idx] = fade_out_values[i]

        processed_feature = self.create_processed_feature(feature, interpolated, "Interpolated", invert_output)
        return (processed_feature,)

@apply_tooltips
class FeatureInterpolator(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "interpolation_method": (["zero", "linear", "cubic", "nearest", "previous", "next", "quadratic"],),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_difference": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_distance": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "extrapolate": ("BOOLEAN", {"default": False}),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "modulate"

    def modulate(self, feature, interpolation_method, threshold, min_difference, min_distance, extrapolate, invert_output):
        # Get feature values
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Find significant points based on threshold and difference
        significant_indices = []
        last_value = None
        last_index = None
        
        for i, v in enumerate(values):
            # Check threshold
            if v >= threshold:
                # Check minimum difference from last point
                if last_value is None or abs(v - last_value) >= min_difference:
                    # Check minimum distance from last point
                    if last_index is None or (i - last_index) >= min_distance:
                        significant_indices.append(i)
                        last_value = v
                        last_index = i
        
        if not significant_indices:
            # If no significant points found, return original feature
            return (feature,)
            
        significant_values = [values[i] for i in significant_indices]
        
        # Create interpolation function
        x = np.array(significant_indices)
        y = np.array(significant_values)
        
        # Handle extrapolation
        fill_value = "extrapolate" if extrapolate else (significant_values[0], significant_values[-1])
        
        # Create interpolator based on method
        if interpolation_method == "zero":
            # Zero-order hold (step function)
            f = interp1d(x, y, kind='zero', bounds_error=False, fill_value=fill_value)
        elif interpolation_method == "linear":
            f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=fill_value)
        elif interpolation_method == "cubic":
            # Need at least 4 points for cubic, fallback to quadratic if not enough points
            if len(x) >= 4:
                f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=fill_value)
            else:
                f = interp1d(x, y, kind='quadratic', bounds_error=False, fill_value=fill_value)
        elif interpolation_method == "nearest":
            f = interp1d(x, y, kind='nearest', bounds_error=False, fill_value=fill_value)
        elif interpolation_method == "previous":
            f = interp1d(x, y, kind='previous', bounds_error=False, fill_value=fill_value)
        elif interpolation_method == "next":
            f = interp1d(x, y, kind='next', bounds_error=False, fill_value=fill_value)
        elif interpolation_method == "quadratic":
            # Need at least 3 points for quadratic, fallback to linear if not enough points
            if len(x) >= 3:
                f = interp1d(x, y, kind='quadratic', bounds_error=False, fill_value=fill_value)
            else:
                f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=fill_value)
        
        # Generate interpolated values for all frames
        x_new = np.arange(feature.frame_count)
        interpolated = f(x_new)
        
        # Create processed feature
        processed_feature = self.create_processed_feature(feature, interpolated, "Interpolated", invert_output)
        return (processed_feature,)

@apply_tooltips
class FeaturePeakDetector(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "prominence": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "distance": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "width": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "plateau_size": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "detect_valleys": ("BOOLEAN", {"default": False}),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("FEATURE",)
    RETURN_NAMES = ("FEATURE",)
    FUNCTION = "modulate"

    def modulate(self, feature, prominence, distance, width, plateau_size, detect_valleys, invert_output):
        # Get feature values
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Convert to numpy array for processing
        signal = np.array(values)
        
        # If detecting valleys, invert the signal temporarily
        if detect_valleys:
            signal = -signal
            
        # Find peaks with given parameters
        peaks, properties = find_peaks(
            signal,
            prominence=prominence,  # Minimum prominence of peaks
            distance=distance,      # Minimum distance between peaks
            width=width,           # Minimum width of peaks
            plateau_size=plateau_size  # Minimum size of flat peaks
        )
        
        # Create output signal where peaks are 1.0 and everything else is 0.0
        peak_signal = np.zeros_like(signal)
        peak_signal[peaks] = 1.0
        
        # Create processed feature
        name_prefix = "Valleys" if detect_valleys else "Peaks"
        processed_feature = self.create_processed_feature(feature, peak_signal, name_prefix, invert_output)
        return (processed_feature,)

