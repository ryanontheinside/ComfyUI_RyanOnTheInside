import torch
import comfy.model_management
import comfy.samplers
from .ace_step_guiders import ACEStepRepaintGuider, ACEStepExtendGuider, ACEStepHybridGuider
from .ace_step_utils import ACEStepLatentUtils
from .audio_mask_nodes import (
    AudioTemporalMask, AudioRegionMask, AudioMaskAnalyzer,
    AUDIO_MASK_NODE_CLASS_MAPPINGS, AUDIO_MASK_NODE_DISPLAY_NAME_MAPPINGS
)

def validate_audio_latent(latents):
    """Validate that latents are audio latents, handling both typed and untyped cases"""
    # Check if it's explicitly marked as audio type
    if latents.get("type") == "audio":
        return True
    
    # If no type field, check if it has the ACE audio latent shape (batch, 8, 16, length)
    if "samples" in latents:
        tensor = latents["samples"]
        if len(tensor.shape) == 4 and tensor.shape[1] == 8 and tensor.shape[2] == 16:
            return True
    
    return False

class ACEStepRepaintGuiderNode:
    """Node that creates a repaint guider for use with SamplerCustomAdvanced"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "source_latents": ("LATENT",),
                "start_time": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "repaint_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_time": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"
    
    def get_guider(self, model, positive, negative, cfg, source_latents, 
                  start_time, end_time, repaint_strength, feather_time):
        
        # Validate inputs
        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")
        
        if not validate_audio_latent(source_latents):
            raise ValueError("source_latents must be audio latents (from VAEEncodeAudio or EmptyAceStepLatentAudio)")
        
        # Extract latent tensor
        latent_tensor = source_latents["samples"]
        
        # Create and return the repaint guider
        guider = ACEStepRepaintGuider(
            model, positive, negative, cfg,
            latent_tensor, start_time, end_time, 
            repaint_strength, feather_time
        )
        
        return (guider,)

class ACEStepExtendGuiderNode:
    """Node that creates an extend guider for use with SamplerCustomAdvanced"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "source_latents": ("LATENT",),
                "extend_left_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "extend_right_time": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"
    
    def get_guider(self, model, positive, negative, cfg, source_latents,
                  extend_left_time, extend_right_time):
        
        if extend_left_time == 0 and extend_right_time == 0:
            raise ValueError("At least one of extend_left_time or extend_right_time must be > 0")
        
        if not validate_audio_latent(source_latents):
            raise ValueError("source_latents must be audio latents (from VAEEncodeAudio or EmptyAceStepLatentAudio)")
        
        # Extract latent tensor
        latent_tensor = source_latents["samples"]
        
        # Create and return the extend guider
        guider = ACEStepExtendGuider(
            model, positive, negative, cfg,
            latent_tensor, extend_left_time, extend_right_time
        )
        
        return (guider,)

class ACEStepHybridGuiderNode:
    """Node that creates a hybrid guider supporting both repaint and extend"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "source_latents": ("LATENT",),
                "extend_left_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "extend_right_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "repaint_start_time": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
                "repaint_end_time": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
                "repaint_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feather_time": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"
    
    def get_guider(self, model, positive, negative, cfg, source_latents,
                  extend_left_time, extend_right_time,
                  repaint_start_time=-1.0, repaint_end_time=-1.0, 
                  repaint_strength=0.7, feather_time=0.1):
        
        if not validate_audio_latent(source_latents):
            raise ValueError("source_latents must be audio latents (from VAEEncodeAudio or EmptyAceStepLatentAudio)")
        
        # Check if repaint is enabled (negative values disable it)
        has_repaint = repaint_start_time >= 0 and repaint_end_time >= 0
        has_extend = extend_left_time > 0 or extend_right_time > 0
        
        if not has_repaint and not has_extend:
            raise ValueError("Must enable either repaint (set repaint times >= 0) or extend (set extend times > 0)")
        
        if has_repaint and repaint_start_time >= repaint_end_time:
            raise ValueError(f"repaint_start_time ({repaint_start_time}) must be less than repaint_end_time ({repaint_end_time})")
        
        # Extract latent tensor
        latent_tensor = source_latents["samples"]
        
        # Create and return the hybrid guider
        guider = ACEStepHybridGuider(
            model, positive, negative, cfg, latent_tensor,
            repaint_start_time if has_repaint else None,
            repaint_end_time if has_repaint else None,
            repaint_strength, feather_time,
            extend_left_time, extend_right_time
        )
        
        return (guider,)

class ACEStepAnalyzeLatentNode:
    """Node that analyzes ACE latent properties for debugging and information"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("info", "duration_seconds", "frame_count", "channels")
    FUNCTION = "analyze_latent"
    CATEGORY = "latent/audio"
    
    def analyze_latent(self, latents):
        if not validate_audio_latent(latents):
            return ("Not audio latents", 0.0, 0, 0)
        
        latent_tensor = latents["samples"]
        batch_size, channels, height, length = latent_tensor.shape
        
        # Calculate duration
        duration = ACEStepLatentUtils.frame_index_to_time(length)
        
        # Create info string
        info = f"ACE Audio Latent Analysis:\n"
        info += f"Shape: {latent_tensor.shape}\n"
        info += f"Batch size: {batch_size}\n"
        info += f"Channels: {channels}\n"
        info += f"Height: {height}\n"
        info += f"Length (frames): {length}\n"
        info += f"Duration: {duration:.2f} seconds\n"
        info += f"Frame rate: ~{length/duration:.1f} frames/second"
        
        return (info, duration, length, channels)

class ACEStepTimeRangeNode:
    """Node for converting time ranges to frame indices for debugging"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("info", "start_frame", "end_frame")
    FUNCTION = "convert_time_range"
    CATEGORY = "latent/audio"
    
    def convert_time_range(self, start_time, end_time):
        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")
        
        start_frame = ACEStepLatentUtils.time_to_frame_index(start_time)
        end_frame = ACEStepLatentUtils.time_to_frame_index(end_time)
        
        info = f"Time Range Conversion:\n"
        info += f"Start: {start_time}s → Frame {start_frame}\n"
        info += f"End: {end_time}s → Frame {end_frame}\n"
        info += f"Duration: {end_time - start_time}s ({end_frame - start_frame} frames)"
        
        return (info, start_frame, end_frame)

class ACEStepMaskVisualizerNode:
    """Node for visualizing repaint/extend masks (for debugging)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT",),
                "start_time": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "end_time": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "feather_time": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mask_info",)
    FUNCTION = "visualize_mask"
    CATEGORY = "latent/audio"
    
    def visualize_mask(self, latents, start_time, end_time, feather_time):
        if not validate_audio_latent(latents):
            return ("Not audio latents",)
        
        latent_tensor = latents["samples"]
        
        # Create mask
        start_frame = ACEStepLatentUtils.time_to_frame_index(start_time)
        end_frame = ACEStepLatentUtils.time_to_frame_index(end_time)
        feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_time)
        
        mask = ACEStepLatentUtils.create_repaint_mask(latent_tensor.shape, start_frame, end_frame)
        
        if feather_frames > 0:
            feathered_mask = ACEStepLatentUtils.create_feather_mask(mask, feather_frames)
        else:
            feathered_mask = mask
        
        # Create visualization
        total_frames = latent_tensor.shape[-1]
        mask_summary = feathered_mask[0, 0, 0, :].cpu().numpy()  # Take first channel of first batch
        
        info = f"Mask Visualization (Length: {total_frames} frames):\n"
        info += f"Repaint region: Frame {start_frame} to {end_frame}\n"
        info += f"Feather frames: {feather_frames}\n\n"
        
        # Create simple ASCII visualization
        viz_length = min(80, total_frames)  # Limit to 80 characters
        scale = total_frames / viz_length
        
        viz_line = ""
        for i in range(viz_length):
            frame_idx = int(i * scale)
            mask_val = mask_summary[frame_idx] if frame_idx < len(mask_summary) else 0
            if mask_val > 0.8:
                viz_line += "█"
            elif mask_val > 0.6:
                viz_line += "▓"
            elif mask_val > 0.4:
                viz_line += "▒"
            elif mask_val > 0.2:
                viz_line += "░"
            else:
                viz_line += "·"
        
        info += f"Mask pattern: {viz_line}\n"
        info += "Legend: █=1.0, ▓=0.8, ▒=0.6, ░=0.4, ·=0.2 or less"
        
        return (info,)

class ACEStepAudioPostProcessor:
    """Post-processes audio to restore original fidelity in preserved regions"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_audio": ("AUDIO",),
                "processed_audio": ("AUDIO",),
                "guider": ("GUIDER",),
                "crossfade_duration": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "post_process_audio"
    CATEGORY = "ACEStep Native"
    
    def post_process_audio(self, original_audio, processed_audio, guider, crossfade_duration):
        """
        Post-process audio to restore original fidelity in preserved regions
        
        Args:
            original_audio: Original input audio (before VAE encoding/decoding)
            processed_audio: Audio output from ACEStep processing
            guider: The guider object containing mask information
            crossfade_duration: Duration in seconds for crossfading at boundaries
        """
        import torch
        import torchaudio.functional as F
        from .ace_step_utils import ACEStepLatentUtils
        
        
        # Extract audio tensors and sample rates from dictionary format
        orig_waveform = original_audio['waveform']
        orig_sr = original_audio['sample_rate']
        proc_waveform = processed_audio['waveform']
        proc_sr = processed_audio['sample_rate']
        
        
        # Ensure sample rates match
        if orig_sr != proc_sr:
            orig_waveform = F.resample(orig_waveform, orig_sr, proc_sr)
            orig_sr = proc_sr
        
        # Handle length differences between original and processed audio
        orig_length = orig_waveform.shape[-1]
        proc_length = proc_waveform.shape[-1]
        
        if proc_length > orig_length:
            # Extend case: processed audio is longer, need to pad original audio
            
            # Check if this is an extend guider to determine padding strategy
            if hasattr(guider, 'left_frames') and hasattr(guider, 'right_frames'):
                # We know the extend amounts, pad accordingly
                left_samples = int(guider.left_frames * 512 * 8 / 44100 * orig_sr)
                right_samples = proc_length - orig_length - left_samples
                
                
                # Pad original audio to match processed length
                orig_waveform = torch.nn.functional.pad(
                    orig_waveform, (left_samples, right_samples), "constant", 0.0
                )
            else:
                # Fallback: pad equally on both sides
                total_padding = proc_length - orig_length
                left_padding = total_padding // 2
                right_padding = total_padding - left_padding
                
                
                orig_waveform = torch.nn.functional.pad(
                    orig_waveform, (left_padding, right_padding), "constant", 0.0
                )
                
        elif orig_length > proc_length:
            # Processed audio is shorter, pad it (shouldn't happen in normal extend/repaint)
            proc_waveform = torch.nn.functional.pad(
                proc_waveform, (0, orig_length - proc_length), "constant", 0.0
            )
        
        # Now both should be the same length
        final_length = max(orig_waveform.shape[-1], proc_waveform.shape[-1])
        
        # Create audio-space mask from guider
        audio_mask = self._create_audio_mask_from_guider(guider, final_length, orig_sr)
        
        
        # Apply crossfading if requested
        if crossfade_duration > 0:
            crossfade_samples = int(crossfade_duration * orig_sr)
            audio_mask = self._apply_crossfade_to_mask(audio_mask, crossfade_samples)
        
        # Blend audio: use original where mask=0, processed where mask=1, smooth transition in between
        result_waveform = audio_mask * proc_waveform + (1 - audio_mask) * orig_waveform
        
        
        # Calculate preservation statistics
        preserved_ratio = (1 - audio_mask.mean().item())
        
        # Return in the same dictionary format
        return ({"waveform": result_waveform, "sample_rate": orig_sr},)
    
    def _create_audio_mask_from_guider(self, guider, audio_length, sample_rate):
        """Create audio-space mask from guider's latent-space mask"""
        import torch
        from .ace_step_utils import ACEStepLatentUtils
        
        # Get the appropriate mask from the guider
        if hasattr(guider, 'combined_mask') and guider.combined_mask is not None:
            # Hybrid guider
            latent_mask = guider.combined_mask
        elif hasattr(guider, 'repaint_mask') and guider.repaint_mask is not None:
            # Repaint guider
            latent_mask = guider.repaint_mask
        elif hasattr(guider, 'extend_mask') and guider.extend_mask is not None:
            # Extend guider
            latent_mask = guider.extend_mask
        else:
            # Fallback: no mask, preserve everything
            return torch.zeros(1, audio_length)
        
        # Convert latent mask to audio mask
        # Latent frames correspond to audio segments
        latent_frames = latent_mask.shape[-1]
        audio_frames_per_latent = audio_length // latent_frames
        
        # Upsample mask to audio resolution
        # Take the first channel/batch element for the mask pattern
        mask_pattern = latent_mask[0, 0, 0, :].cpu()  # Shape: [latent_frames]
        
        # Repeat each latent frame mask value for corresponding audio samples
        audio_mask = mask_pattern.repeat_interleave(audio_frames_per_latent)
        
        # Handle any remaining samples if audio_length isn't perfectly divisible
        if len(audio_mask) < audio_length:
            # Pad with the last mask value
            padding = torch.full((audio_length - len(audio_mask),), mask_pattern[-1])
            audio_mask = torch.cat([audio_mask, padding])
        elif len(audio_mask) > audio_length:
            # Trim to exact length
            audio_mask = audio_mask[:audio_length]
        
        # Reshape to match audio tensor format [channels, samples]
        audio_mask = audio_mask.unsqueeze(0)  # Add channel dimension
        
        return audio_mask
    
    def _apply_crossfade_to_mask(self, mask, crossfade_samples):
        """Apply smooth crossfading at mask boundaries to avoid audio artifacts"""
        import torch
        
        if crossfade_samples <= 0:
            return mask
        
        # Create smoothed mask
        smoothed_mask = mask.clone()
        
        # Find transitions in the mask (0->1 and 1->0)
        mask_1d = mask[0]  # Remove channel dimension for processing
        diff = torch.diff(mask_1d, prepend=mask_1d[0:1])
        
        # Find rising edges (0->1 transitions) - start of generated regions
        rising_edges = torch.where(diff > 0.5)[0]
        # Find falling edges (1->0 transitions) - end of generated regions  
        falling_edges = torch.where(diff < -0.5)[0]
        
        
        # Apply fade-in at rising edges
        for edge in rising_edges:
            start_idx = max(0, edge - crossfade_samples // 2)
            end_idx = min(len(mask_1d), edge + crossfade_samples // 2)
            fade_length = end_idx - start_idx
            if fade_length > 0:
                fade_in = torch.linspace(0, 1, fade_length)
                smoothed_mask[0, start_idx:end_idx] = torch.maximum(
                    smoothed_mask[0, start_idx:end_idx], fade_in
                )
        
        # Apply fade-out at falling edges
        for edge in falling_edges:
            start_idx = max(0, edge - crossfade_samples // 2)
            end_idx = min(len(mask_1d), edge + crossfade_samples // 2)
            fade_length = end_idx - start_idx
            if fade_length > 0:
                fade_out = torch.linspace(1, 0, fade_length)
                smoothed_mask[0, start_idx:end_idx] = torch.minimum(
                    smoothed_mask[0, start_idx:end_idx], fade_out
                )
        
        return smoothed_mask

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ACEStepRepaintGuider": ACEStepRepaintGuiderNode,
    "ACEStepExtendGuider": ACEStepExtendGuiderNode,
    "ACEStepHybridGuider": ACEStepHybridGuiderNode,
    "ACEStepAnalyzeLatent": ACEStepAnalyzeLatentNode,
    "ACEStepTimeRange": ACEStepTimeRangeNode,
    "ACEStepMaskVisualizer": ACEStepMaskVisualizerNode,
    "ACEStepAudioPostProcessor": ACEStepAudioPostProcessor,
    **AUDIO_MASK_NODE_CLASS_MAPPINGS,  # Add audio mask nodes
}

# Display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ACEStepRepaintGuider": "ACEStep Repaint Guider BETA",
    "ACEStepExtendGuider": "ACEStep Extend Guider BETA", 
    "ACEStepHybridGuider": "ACEStep Hybrid Guider BETA",
    "ACEStepAnalyzeLatent": "ACEStep Analyze Latent BETA",
    "ACEStepTimeRange": "ACEStep Time Range BETA",
    "ACEStepMaskVisualizer": "ACEStep Mask Visualizer BETA",
    "ACEStepAudioPostProcessor": "ACEStep Audio Post Processor BETA",
    **AUDIO_MASK_NODE_DISPLAY_NAME_MAPPINGS,  # Add audio mask node display names
} 