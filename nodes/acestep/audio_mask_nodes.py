import torch
import comfy.model_management
from .ace_step_utils import ACEStepLatentUtils

class AudioMaskBase:
    """Base class for audio mask nodes with common functionality"""
    
    def _extract_and_validate_latent(self, audio_latents):
        """Extract and validate ACE audio latent tensor"""
        if "samples" in audio_latents:
            latent_tensor = audio_latents["samples"]
        else:
            raise ValueError("Invalid latent format")
        
        ACEStepLatentUtils.validate_ace_latent_shape(latent_tensor)
        return latent_tensor
    
    def _create_base_mask(self, latent_tensor):
        """Create base mask with correct shape for CondPairSetProps"""
        batch_size, channels, height, length = latent_tensor.shape
        return torch.zeros((batch_size, height, length)), (batch_size, height, length)
    
    def _finalize_mask(self, mask):
        """Ensure mask values are in valid range"""
        return torch.clamp(mask, 0.0, 1.0)

class AudioTemporalMask(AudioMaskBase):
    """Create temporal crossfade mask from a list of float values"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "length_mismatch": (["repeat", "loop"], {"default": "repeat"}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_temporal_mask"
    CATEGORY = "conditioning/audio"
    
    def create_temporal_mask(self, audio_latents, value, length_mismatch):
        """Create temporal mask from custom float values"""
        
        latent_tensor = self._extract_and_validate_latent(audio_latents)
        mask, (batch_size, height, length) = self._create_base_mask(latent_tensor)
        
        # Handle both single float and list of floats
        if isinstance(value, (int, float)):
            # Single value - create list with just that value
            values = [float(value)]
        elif isinstance(value, (list, tuple)):
            # List/tuple of values
            values = [float(x) for x in value]
        else:
            raise ValueError(f"float_values must be a float or list of floats, got {type(value)}")
        
        if not values:
            raise ValueError("No valid float values provided")
        
        # Handle length mismatch
        if len(values) > length:
            # Truncate if too long
            values = values[:length]
        elif len(values) < length:
            # Extend if too short
            if length_mismatch == "repeat":
                # Repeat last value
                last_value = values[-1]
                values.extend([last_value] * (length - len(values)))
            else:  # loop
                # Loop from beginning
                while len(values) < length:
                    remaining = length - len(values)
                    values.extend(values[:min(remaining, len(values))])
        
        # Apply values to mask
        for i in range(length):
            mask[:, :, i] = values[i]
        
        return (self._finalize_mask(mask),)

class AudioRegionMask(AudioMaskBase):
    """Create masks for specific time regions - useful for section-based conditioning"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "start_time": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "mask_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_region_mask"
    CATEGORY = "conditioning/audio"
    
    def create_region_mask(self, audio_latents, start_time, end_time, mask_value, feather_seconds):
        """Create mask for specific time region with optional feathering"""
        
        latent_tensor = self._extract_and_validate_latent(audio_latents)
        mask, (batch_size, height, length) = self._create_base_mask(latent_tensor)
        
        # Convert to frames
        start_frame = ACEStepLatentUtils.time_to_frame_index(start_time)
        end_frame = ACEStepLatentUtils.time_to_frame_index(end_time)
        feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_seconds)
        
        # Set region value
        if start_frame < end_frame:
            mask[:, :, start_frame:end_frame] = mask_value
        
        # Apply feathering if requested
        if feather_frames > 0:
            # Feather start
            for i in range(max(0, start_frame - feather_frames), start_frame):
                if i >= 0 and i < length:
                    progress = (i - (start_frame - feather_frames)) / feather_frames
                    mask[:, :, i] = mask_value * progress
            
            # Feather end
            for i in range(end_frame, min(length, end_frame + feather_frames)):
                if i >= 0 and i < length:
                    progress = 1.0 - (i - end_frame) / feather_frames
                    mask[:, :, i] = mask_value * progress
        
        return (self._finalize_mask(mask),)

class AudioMaskAnalyzer:
    """Analyze audio masks - see what you created"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_mask"
    CATEGORY = "conditioning/audio"
    
    def analyze_mask(self, mask):
        """Show mask info and visualization"""
        
        info = f"Audio Mask Analysis:\n"
        info += f"Shape: {mask.shape}\n"
        
        if len(mask.shape) >= 3:
            batch_size, height, length = mask.shape[0], mask.shape[1], mask.shape[-1]
            duration = ACEStepLatentUtils.frame_index_to_time(length)
            
            info += f"Duration: {duration:.2f} seconds\n"
            info += f"Frames: {length}\n"
            
            # Get mask values along time dimension (first batch, first height)
            mask_values = mask[0, 0, :].cpu().numpy()
            
            info += f"Value range: {mask_values.min():.3f} to {mask_values.max():.3f}\n"
            
            # Create simple visualization
            viz_length = min(60, length)
            step = length / viz_length if viz_length > 0 else 1
            
            viz_line = ""
            for i in range(viz_length):
                idx = int(i * step)
                val = mask_values[idx] if idx < len(mask_values) else 0
                
                if val > 0.8:
                    viz_line += "█"
                elif val > 0.6:
                    viz_line += "▓"
                elif val > 0.4:
                    viz_line += "▒"
                elif val > 0.2:
                    viz_line += "░"
                else:
                    viz_line += "·"
            
            info += f"\nTemporal pattern:\n{viz_line}\n"
            info += "█=0.8+ ▓=0.6+ ▒=0.4+ ░=0.2+ ·=<0.2\n"
            info += f"Time: 0s{'':>{viz_length-10}}{duration:.1f}s"
        
        return (info,)

class MaskToAudioMask(AudioMaskBase):
    """Convert spatial masks (from video frames) to audio temporal masks - batch dimension = time"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "spatial_mask": ("MASK",),
                "frame_summary": (["average", "max", "min", "center_pixel"], {"default": "average"}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert_spatial_to_audio_mask"
    CATEGORY = "conditioning/audio"
    
    def convert_spatial_to_audio_mask(self, audio_latents, spatial_mask, frame_summary):
        """Convert spatial mask to audio temporal mask - batch=time, spatial→single value per frame"""
        
        latent_tensor = self._extract_and_validate_latent(audio_latents)
        audio_mask, (batch_size, height, target_length) = self._create_base_mask(latent_tensor)
        
        # Handle different spatial mask shapes
        if len(spatial_mask.shape) == 2:
            # (height, width) -> single frame, add batch dimension
            spatial_mask = spatial_mask.unsqueeze(0)
        elif len(spatial_mask.shape) == 4:
            # (batch, channels, height, width) -> take first channel
            spatial_mask = spatial_mask[:, 0, :, :]
        # Now: (frames, height, width) where frames = time
        
        num_frames = spatial_mask.shape[0]
        
        # Extract temporal pattern: each frame → single value
        temporal_values = []
        
        for frame_idx in range(num_frames):
            frame = spatial_mask[frame_idx]  # (height, width)
            
            if frame_summary == "average":
                # Average all pixels in frame
                value = frame.mean().item()
            elif frame_summary == "max":
                # Max pixel value in frame
                value = frame.max().item()
            elif frame_summary == "min":
                # Min pixel value in frame
                value = frame.min().item()
            elif frame_summary == "center_pixel":
                # Center pixel value
                h, w = frame.shape
                value = frame[h//2, w//2].item()
            
            temporal_values.append(value)
        
        # Interpolate temporal pattern to match audio length
        if len(temporal_values) != target_length:
            # Convert to tensor for interpolation
            pattern_tensor = torch.tensor(temporal_values, dtype=torch.float32)
            pattern_2d = pattern_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, frames)
            
            # Interpolate to target audio length
            resized = torch.nn.functional.interpolate(
                pattern_2d, size=target_length, mode='linear', align_corners=False
            )
            final_pattern = resized[0, 0]  # (target_length,)
        else:
            final_pattern = torch.tensor(temporal_values, dtype=torch.float32)
        
        # Apply temporal pattern to all batches and height channels
        for b in range(batch_size):
            for h in range(height):
                audio_mask[b, h, :] = final_pattern
        
        return (self._finalize_mask(audio_mask),)

class AudioLatentInfo(AudioMaskBase):
    """Extract dimensional info from audio latents for use with other nodes"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "frame_rate": ("FLOAT", {"default": 10.77, "min": 0.1, "max": 100.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("batch_size", "length_frames", "duration_seconds", "info")
    FUNCTION = "get_audio_info"
    CATEGORY = "conditioning/audio"
    
    def get_audio_info(self, audio_latents, frame_rate):
        """Extract batch size, length, and duration from audio latents"""
        
        latent_tensor = self._extract_and_validate_latent(audio_latents)
        batch_size, channels, height, actual_length = latent_tensor.shape
        
        # Calculate duration using the default frame rate (10.77)
        default_frame_rate = 10.77
        duration = actual_length / default_frame_rate
        
        # Calculate length_frames based on the specified frame rate
        length_frames = int(duration * frame_rate)
        
        # Create info string
        info = f"Audio Latent Info:\n"
        info += f"Batch size: {batch_size}\n"
        info += f"Channels: {channels}\n"
        info += f"Height: {height}\n"
        info += f"Actual length: {actual_length} frames\n"
        info += f"Frame rate: {frame_rate:.2f} fps\n"
        info += f"Length: {length_frames} frames\n"
        info += f"Duration: {duration:.2f} seconds"
        
        return (batch_size, length_frames, duration, info)

# Node mappings
AUDIO_MASK_NODE_CLASS_MAPPINGS = {
    "AudioTemporalMask": AudioTemporalMask,
    "AudioRegionMask": AudioRegionMask,
    "AudioMaskAnalyzer": AudioMaskAnalyzer,
    "MaskToAudioMask": MaskToAudioMask,
    "AudioLatentInfo": AudioLatentInfo,
}

AUDIO_MASK_NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTemporalMask": "Audio Temporal Mask",
    "AudioRegionMask": "Audio Region Mask", 
    "AudioMaskAnalyzer": "Audio Mask Analyzer",
    "MaskToAudioMask": "Mask to Audio Mask",
    "AudioLatentInfo": "Audio Latent Info",
} 