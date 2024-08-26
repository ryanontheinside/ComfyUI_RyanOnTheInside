from ... import RyanOnTheInside
import numpy as np
import matplotlib.pyplot as plt
import torch
from io import BytesIO
from PIL import Image

class FeatureModulationBase(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexFeatures"
    FUNCTION = "modulate"
    
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
                "rise_detection_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rise_smoothing_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")
    FUNCTION = "modulate"

    def modulate(self, feature, base_gain, floor, ceiling, peak_sharpness, valley_sharpness, attack, release, smoothing, feature_threshold, rise_detection_threshold, rise_smoothing_factor):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # feature threshold
        values = [v if v >= feature_threshold else 0 for v in values]
        
        # 0-1 range
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            normalized = [0 for _ in values]  # All values are the same, normalize to 0
        else:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        
        # multiply gain
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
        
        #we create a new feature with our values and override extract method
        # NOTE: tested with several feature type, but not all
        class ProcessedFeature(type(feature)):
            def __init__(self, original_feature, processed_values):
                self.__dict__.update(original_feature.__dict__)
                self.name = f"Processed_{original_feature.name}"
                self.frame_rate = original_feature.frame_rate
                self.frame_count = original_feature.frame_count
                self.data = processed_values

            def extract(self):
                return self

            def get_value_at_frame(self, frame_index):
                return self.data[frame_index]

        processed_feature = ProcessedFeature(feature, final_values)
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