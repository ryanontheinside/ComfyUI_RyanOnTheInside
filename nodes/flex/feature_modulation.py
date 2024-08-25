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
        
        # Use BytesIO to store the image data
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='black', edgecolor='none')
        buf.seek(0)
        
        # Open the image with PIL
        img = Image.open(buf)
        
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Convert to torch tensor and normalize
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        
        # Add batch dimension if not present
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        plt.close()  # Close the plot to free up memory
        buf.close()  # Close the BytesIO buffer
        
        return img_tensor

class FeatureStudio(FeatureModulationBase):
    @classmethod
    def INPUT_TYPES(s):
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
            }
        }

    RETURN_TYPES = ("FEATURE", "IMAGE")
    RETURN_NAMES = ("FEATURE", "FEATURE_VISUALIZATION")

    def modulate(self, feature, base_gain, floor, ceiling, peak_sharpness, valley_sharpness, attack, release, smoothing):
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Normalize values to 0-1 range
        min_val, max_val = min(values), max(values)
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
        
        # Apply base gain
        gained = [v * base_gain for v in normalized]
        
        # Apply waveshaping (asymmetric sharpness)
        def waveshape(v):
            return v**peak_sharpness if v > 0.5 else 1 - (1-v)**valley_sharpness
        waveshaped = [waveshape(v) for v in gained]
        
        # Apply envelope follower
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
        
        # Apply smoothing
        def smooth_values(values, smoothing_factor):
            smoothed = values.copy()
            for i in range(1, len(values)):
                smoothed[i] = smoothed[i-1] + (values[i] - smoothed[i-1]) * (1 - smoothing_factor)
            return smoothed
        smoothed = smooth_values(enveloped, smoothing)
        
        # Apply floor and ceiling
        final_values = [max(floor, min(ceiling, v)) for v in smoothed]
        
        # Create a new feature with the processed values
        class ProcessedFeature(type(feature)):
            def __init__(self, original_feature, processed_values):
                super().__init__(original_feature.name, original_feature.frame_rate, original_feature.frame_count)
                self.processed_values = processed_values

            def get_value_at_frame(self, frame_index):
                return self.processed_values[frame_index]

        processed_feature = ProcessedFeature(feature, final_values)
        return (processed_feature, self.visualize(processed_feature))