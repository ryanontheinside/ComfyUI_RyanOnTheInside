from ... import RyanOnTheInside
import numpy as np
import matplotlib.pyplot as plt
import torch
from io import BytesIO
from PIL import Image

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
    
    def visualize(self, feature, width=1920, height=1080):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        frames = len(values)
        
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.style.use('dark_background')
        
        plt.plot(values, color='dodgerblue', linewidth=2)
        
        plt.xlabel('Frame', color='white', fontsize=14)
        plt.ylabel('Value', color='white', fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        plt.tick_params(axis='both', colors='white', labelsize=12)
        
        max_ticks = 10
        step = max(1, frames // max_ticks)
        x_ticks = range(0, frames, step)
        plt.xticks(x_ticks, [str(x) for x in x_ticks])
        
        y_min, y_max = min(values), max(values)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        
        plt.title(f'Feature: {feature.name}', color='white', fontsize=16)
        
        plt.tight_layout(pad=0.5)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='black', edgecolor='none')
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        plt.close()  
        buf.close()  
        
        return img_tensor

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
                "rise_smoothing_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                **super().INPUT_TYPES()["required"],  # Include the invert_output option
            }
        }

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")
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
        return (processed_feature, self.visualize(processed_feature))

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

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")

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
        return (processed_feature, self.visualize(processed_feature))

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

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")

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
        return (processed_feature, self.visualize(processed_feature))

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

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")
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
        return (processed_feature, self.visualize(processed_feature))
    
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

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")

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
        return (processed_feature, self.visualize(processed_feature))

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

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")

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
        return (processed_feature, self.visualize(processed_feature))


#NOTE  separated from FeatureMath for ease  of  use.
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

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")

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
        return (processed_feature, self.visualize(processed_feature))

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

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")
    FUNCTION = "rebase"

    def rebase(self, feature, lower_threshold, upper_threshold, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Apply thresholds
        rebased_values = [v if lower_threshold <= v <= upper_threshold else 0 for v in values]
        
        # Re-normalize the values
        min_val, max_val = min(rebased_values), max(rebased_values)
        if min_val == max_val:
            normalized = [0 for _ in rebased_values]  # All values are the same, normalize to 0
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in rebased_values]
        
        processed_feature = self.create_processed_feature(feature, normalized, "Rebased", invert_output)
        return (processed_feature, self.visualize(processed_feature))

class PreviewFeature(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                **super().INPUT_TYPES()["required"],
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("FEATURE_PREVIEW",)
    FUNCTION = "preview"

    def preview(self, feature, invert_output):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        if invert_output:
            min_val, max_val = min(values), max(values)
            inverted_values = [max_val - v + min_val for v in values]
            processed_feature = self.create_processed_feature(feature, inverted_values, "Inverted", invert_output)
        else:
            processed_feature = feature
        
        return (self.visualize(processed_feature),)