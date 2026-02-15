import torch
import comfy.model_management
from .ace_step_utils import ACEStepLatentUtils

class AudioMaskBase:
    """Base class for audio mask nodes with common functionality"""

    def _extract_and_validate_latent(self, audio_latents):
        """Extract and validate ACE audio latent tensor, returns (tensor, version)"""
        if "samples" in audio_latents:
            latent_tensor = audio_latents["samples"]
        else:
            raise ValueError("Invalid latent format")

        version = ACEStepLatentUtils.validate_ace_latent_shape(latent_tensor)
        return latent_tensor, version

    def _create_base_mask(self, latent_tensor, version=None):
        """Create base mask with correct shape for conditioning
        v1.0: (batch, height, length) where height=16 - matches noise.shape[2:] = (16, length)
        v1.5: (batch, length) - matches noise.shape[2:] = (length,)
        """
        if version == ACEStepLatentUtils.V1_5:
            batch_size, channels, length = latent_tensor.shape
            mask = torch.zeros((batch_size, length))
            dims = (batch_size, length)
            return mask, dims
        else:
            batch_size, channels, height, length = latent_tensor.shape
            mask = torch.zeros((batch_size, height, length))
            dims = (batch_size, height, length)
            return mask, dims

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

        latent_tensor, version = self._extract_and_validate_latent(audio_latents)
        mask, dims = self._create_base_mask(latent_tensor, version)
        batch_size = dims[0]
        length = dims[-1]  # Always last dimension
        
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

        # Apply values to mask - v1.5 is 2D (batch, length), v1.0 is 3D (batch, height, length)
        for i in range(length):
            if len(mask.shape) == 2:
                mask[:, i] = values[i]
            else:
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

        latent_tensor, version = self._extract_and_validate_latent(audio_latents)
        mask, dims = self._create_base_mask(latent_tensor, version)
        length = dims[-1]  # Always last dimension

        # Convert to frames using version-specific FPS
        start_frame = ACEStepLatentUtils.time_to_frame_index(start_time, version)
        end_frame = ACEStepLatentUtils.time_to_frame_index(end_time, version)
        feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_seconds, version)
        
        # Set region value - v1.5 is 2D (batch, length), v1.0 is 3D (batch, height, length)
        is_2d = len(mask.shape) == 2
        if start_frame < end_frame:
            if is_2d:
                mask[:, start_frame:end_frame] = mask_value
            else:
                mask[:, :, start_frame:end_frame] = mask_value

        # Apply feathering if requested
        if feather_frames > 0:
            # Feather start
            for i in range(max(0, start_frame - feather_frames), start_frame):
                if i >= 0 and i < length:
                    progress = (i - (start_frame - feather_frames)) / feather_frames
                    if is_2d:
                        mask[:, i] = mask_value * progress
                    else:
                        mask[:, :, i] = mask_value * progress

            # Feather end
            for i in range(end_frame, min(length, end_frame + feather_frames)):
                if i >= 0 and i < length:
                    progress = 1.0 - (i - end_frame) / feather_frames
                    if is_2d:
                        mask[:, i] = mask_value * progress
                    else:
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

        if len(mask.shape) >= 2:
            batch_size = mask.shape[0]
            length = mask.shape[-1]

            # Detect version based on shape - v1.5 is 2D, v1.0 is 3D
            if len(mask.shape) == 2:
                version = ACEStepLatentUtils.V1_5
                info += f"Version: v1.5 (2D mask)\n"
            else:
                version = ACEStepLatentUtils.V1_0
                info += f"Version: v1.0 (3D mask)\n"

            duration = ACEStepLatentUtils.frame_index_to_time(length, version)

            info += f"Duration: {duration:.2f} seconds\n"
            info += f"Frames: {length}\n"

            # Get mask values along time dimension
            if len(mask.shape) == 2:
                mask_values = mask[0, :].cpu().numpy()
            else:
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

        latent_tensor, version = self._extract_and_validate_latent(audio_latents)
        audio_mask, dims = self._create_base_mask(latent_tensor, version)
        batch_size = dims[0]
        target_length = dims[-1]
        
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
        
        # Apply temporal pattern to all batches - v1.5 is 2D, v1.0 is 3D
        is_2d = len(audio_mask.shape) == 2
        for b in range(batch_size):
            if is_2d:
                audio_mask[b, :] = final_pattern
            else:
                for h in range(audio_mask.shape[1]):
                    audio_mask[b, h, :] = final_pattern

        return (self._finalize_mask(audio_mask),)

class AudioTemporalMask15(AudioMaskBase):
    """Create temporal crossfade mask for ACE-Step 1.5 latents"""

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
        latent_tensor, version = self._extract_and_validate_latent(audio_latents)
        if version != ACEStepLatentUtils.V1_5:
            raise ValueError("This node is for ACE-Step 1.5. Use AudioTemporalMask for v1.0.")
        mask, dims = self._create_base_mask(latent_tensor, version)
        length = dims[-1]

        if isinstance(value, (int, float)):
            values = [float(value)]
        elif isinstance(value, (list, tuple)):
            values = [float(x) for x in value]
        else:
            raise ValueError(f"float_values must be a float or list of floats, got {type(value)}")

        if not values:
            raise ValueError("No valid float values provided")

        if len(values) > length:
            values = values[:length]
        elif len(values) < length:
            if length_mismatch == "repeat":
                last_value = values[-1]
                values.extend([last_value] * (length - len(values)))
            else:
                while len(values) < length:
                    remaining = length - len(values)
                    values.extend(values[:min(remaining, len(values))])

        for i in range(length):
            mask[:, i] = values[i]

        return (self._finalize_mask(mask),)


class AudioRegionMask15(AudioMaskBase):
    """Create masks for specific time regions for ACE-Step 1.5"""

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
        latent_tensor, version = self._extract_and_validate_latent(audio_latents)
        if version != ACEStepLatentUtils.V1_5:
            raise ValueError("This node is for ACE-Step 1.5. Use AudioRegionMask for v1.0.")
        mask, dims = self._create_base_mask(latent_tensor, version)
        length = dims[-1]

        start_frame = ACEStepLatentUtils.time_to_frame_index(start_time, version)
        end_frame = ACEStepLatentUtils.time_to_frame_index(end_time, version)
        feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_seconds, version)

        if start_frame < end_frame:
            mask[:, start_frame:end_frame] = mask_value

        if feather_frames > 0:
            for i in range(max(0, start_frame - feather_frames), start_frame):
                if i >= 0 and i < length:
                    progress = (i - (start_frame - feather_frames)) / feather_frames
                    mask[:, i] = mask_value * progress
            for i in range(end_frame, min(length, end_frame + feather_frames)):
                if i >= 0 and i < length:
                    progress = 1.0 - (i - end_frame) / feather_frames
                    mask[:, i] = mask_value * progress

        return (self._finalize_mask(mask),)


class MaskToAudioMask15(AudioMaskBase):
    """Convert spatial masks to audio temporal masks for ACE-Step 1.5"""

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
        latent_tensor, version = self._extract_and_validate_latent(audio_latents)
        if version != ACEStepLatentUtils.V1_5:
            raise ValueError("This node is for ACE-Step 1.5. Use MaskToAudioMask for v1.0.")
        mask, dims = self._create_base_mask(latent_tensor, version)
        batch_size = dims[0]
        target_length = dims[-1]

        if len(spatial_mask.shape) == 2:
            spatial_mask = spatial_mask.unsqueeze(0)
        elif len(spatial_mask.shape) == 4:
            spatial_mask = spatial_mask[:, 0, :, :]

        num_frames = spatial_mask.shape[0]
        temporal_values = []
        for frame_idx in range(num_frames):
            frame = spatial_mask[frame_idx]
            if frame_summary == "average":
                value = frame.mean().item()
            elif frame_summary == "max":
                value = frame.max().item()
            elif frame_summary == "min":
                value = frame.min().item()
            elif frame_summary == "center_pixel":
                h, w = frame.shape
                value = frame[h//2, w//2].item()
            temporal_values.append(value)

        if len(temporal_values) != target_length:
            pattern_tensor = torch.tensor(temporal_values, dtype=torch.float32)
            pattern_2d = pattern_tensor.unsqueeze(0).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                pattern_2d, size=target_length, mode='linear', align_corners=False
            )
            final_pattern = resized[0, 0]
        else:
            final_pattern = torch.tensor(temporal_values, dtype=torch.float32)

        for b in range(batch_size):
            mask[b, :] = final_pattern

        return (self._finalize_mask(mask),)


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

        latent_tensor, version = self._extract_and_validate_latent(audio_latents)
        actual_length = latent_tensor.shape[-1]
        batch_size = latent_tensor.shape[0]
        channels = latent_tensor.shape[1]

        # Calculate duration using version-specific frame rate
        native_fps = ACEStepLatentUtils.get_fps(version)
        duration = actual_length / native_fps

        # Calculate length_frames based on the specified frame rate
        length_frames = int(duration * frame_rate)

        # Create info string
        info = f"Audio Latent Info:\n"
        info += f"Version: {version or 'unknown'}\n"
        info += f"Batch size: {batch_size}\n"
        info += f"Channels: {channels}\n"
        if version != ACEStepLatentUtils.V1_5:
            info += f"Height: {latent_tensor.shape[2]}\n"
        info += f"Actual length: {actual_length} frames\n"
        info += f"Native frame rate: {native_fps:.2f} fps\n"
        info += f"Target frame rate: {frame_rate:.2f} fps\n"
        info += f"Length at target rate: {length_frames} frames\n"
        info += f"Duration: {duration:.2f} seconds"

        return (batch_size, length_frames, duration, info)

class AudioLatentStretch(AudioMaskBase):
    """Resize audio latent temporal dimension by a scale factor.
    Values > 1.0 stretch (longer duration), < 1.0 compress (shorter duration).
    Operates in latent space — results may differ from waveform time-stretching."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "latent/audio"

    def upscale(self, audio_latents, scale_by):
        latent_tensor, version = self._extract_and_validate_latent(audio_latents)
        new_length = round(latent_tensor.shape[-1] * scale_by)
        if new_length < 1:
            new_length = 1

        if version == ACEStepLatentUtils.V1_5:
            # [B, C, T] -> interpolate T
            resized = torch.nn.functional.interpolate(
                latent_tensor, size=new_length, mode='linear', align_corners=False
            )
        else:
            # [B, C, H, T] -> interpolate last dim only, preserve H
            b, c, h, t = latent_tensor.shape
            # Reshape to [B*C, H, T] so interpolate treats H as channels
            flat = latent_tensor.reshape(b * c, h, t)
            resized = torch.nn.functional.interpolate(
                flat, size=new_length, mode='linear', align_corners=False
            )
            resized = resized.reshape(b, c, h, new_length)

        s = audio_latents.copy()
        s["samples"] = resized
        return (s,)


# Node mappings
AUDIO_MASK_NODE_CLASS_MAPPINGS = {
    "AudioTemporalMask": AudioTemporalMask,
    "AudioRegionMask": AudioRegionMask,
    "AudioMaskAnalyzer": AudioMaskAnalyzer,
    "MaskToAudioMask": MaskToAudioMask,
    "AudioLatentInfo": AudioLatentInfo,
    "AudioTemporalMask15": AudioTemporalMask15,
    "AudioRegionMask15": AudioRegionMask15,
    "MaskToAudioMask15": MaskToAudioMask15,
    "AudioLatentStretch": AudioLatentStretch,
}

AUDIO_MASK_NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioTemporalMask": "Audio Temporal Mask (ACE 1.0)",
    "AudioRegionMask": "Audio Region Mask (ACE 1.0)",
    "AudioMaskAnalyzer": "Audio Mask Analyzer",
    "MaskToAudioMask": "Mask to Audio Mask (ACE 1.0)",
    "AudioLatentInfo": "Audio Latent Info",
    "AudioTemporalMask15": "Audio Temporal Mask (ACE 1.5)",
    "AudioRegionMask15": "Audio Region Mask (ACE 1.5)",
    "MaskToAudioMask15": "Mask to Audio Mask (ACE 1.5)",
    "AudioLatentStretch": "Audio Latent Stretch",
} 