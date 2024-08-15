import numpy as np
from .mask_base import MaskBase
from .mask_utils import normalize_array, morph_mask

class AudioControlledMaskBase(MaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "audio": ("AUDIO",),
                "bass_freq_range": ("STRING", {"default": "20,150"}),
                "reactivity_curve": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    CATEGORY = "/Masks/Audio"

    def apply_mask_operation(self, masks, audio, bass_freq_range, reactivity_curve, **kwargs):
        audio_np = audio['waveform'].cpu().numpy().squeeze()
        sample_rate = audio['sample_rate']
        
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=0)
        
        num_frames = len(masks)
        samples_per_frame = len(audio_np) // num_frames
        
        frame_energy = np.array([
            np.sqrt(np.mean(audio_np[i*samples_per_frame:(i+1)*samples_per_frame]**2))
            for i in range(num_frames)
        ])
        
        energy_min, energy_max = np.percentile(frame_energy, [5, 95])
        normalized_energy = np.clip((frame_energy - energy_min) / (energy_max - energy_min), 0, 1)
        
        smoothed_energy = np.convolve(normalized_energy, np.ones(5)/5, mode='same')
        enhanced_energy = np.tanh(3 * (smoothed_energy - smoothed_energy.mean()))
        enhanced_energy = normalize_array(enhanced_energy)
        
        enhanced_energy = np.power(enhanced_energy, reactivity_curve)
        enhanced_energy = normalize_array(enhanced_energy)
        
        return super().apply_mask_operation(masks, enhanced_energy, **kwargs)

class AudioControlledMaskMorph(AudioControlledMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "morph_type": (["dilate", "erode", "open", "close"],),
                "kernel_size": ("INT", {"default": 5, "min": 3, "max": 21, "step": 2}),
                "growth_size": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            }
        }

    def process_mask(self, mask, strength, morph_type, kernel_size, growth_size, **kwargs):
        iterations = max(1, int(growth_size * strength * 5))
        return morph_mask(mask, morph_type, kernel_size, iterations)