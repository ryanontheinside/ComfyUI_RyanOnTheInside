import torch
import numpy as np
import comfy.model_management
import comfy.model_sampling
import comfy.samplers

# Import patches module to apply ACE-Step 1.5 patches
from . import patches
from . import logger

from .ace_step_guiders import (
    ACEStepRepaintGuider, ACEStepExtendGuider, ACEStepHybridGuider,
    ACEStep15NativeEditGuider,
    ACEStep15NativeCoverGuider,
    ACEStep15NativeExtractGuider,
    ACEStep15NativeLegoGuider,
)
from .ace_step_utils import ACEStepLatentUtils
from .audio_mask_nodes import (
    AudioTemporalMask, AudioRegionMask, AudioMaskAnalyzer,
    AUDIO_MASK_NODE_CLASS_MAPPINGS, AUDIO_MASK_NODE_DISPLAY_NAME_MAPPINGS
)


def validate_audio_latent(latents):
    """Validate that latents are audio latents, handling both typed and untyped cases
    Supports both v1.0 (batch, 8, 16, length) and v1.5 (batch, 64, length) shapes
    """
    # Check if it's explicitly marked as audio type
    if latents.get("type") == "audio":
        return True

    # If no type field, check shape
    if "samples" in latents:
        tensor = latents["samples"]
        # v1.0: (batch, 8, 16, length)
        if len(tensor.shape) == 4 and tensor.shape[1] == 8 and tensor.shape[2] == 16:
            return True
        # v1.5: (batch, 64, length)
        if len(tensor.shape) == 3 and tensor.shape[1] == 64:
            return True

    return False


class ACEStep15SemanticExtractor:
    """
    Extracts semantic tokens from source audio latents for use with Cover/Extract tasks.

    The ACE-Step 1.5 model uses semantic tokens (lm_hints) as structural guidance
    for tasks like Cover and Extract. This node extracts those tokens from VAE latents.

    Flow: VAE latents → tokenizer → quantized → detokenizer → semantic_hints

    Use the output with ACE-Step 1.5 Cover Guider or Extract Guider for creative
    workflows like blending semantic hints from multiple sources.

    Note: Cover and Extract guiders will auto-extract semantic hints if not provided.
    This node is useful for advanced workflows like semantic hint blending.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "source_latents": ("LATENT",),
            }
        }

    RETURN_TYPES = ("SEMANTIC_HINTS",)
    RETURN_NAMES = ("semantic_hints",)
    FUNCTION = "extract"
    CATEGORY = "audio/acestep"

    def extract(self, model, source_latents):
        from .ace_step_utils import extract_semantic_hints

        source_tensor = source_latents["samples"]
        logger.debug(f"[ACE15_SEMANTIC_EXTRACTOR] Extracting semantic hints from shape {source_tensor.shape}")

        # Use shared utility function
        semantic_hints = extract_semantic_hints(model, source_tensor, verbose=True)

        # Move to CPU for storage
        semantic_hints = semantic_hints.cpu()

        return (semantic_hints,)


class ACEStep15SemanticHintsBlend:
    """
    Blends semantic hints from two sources for creative mashups.

    Use this to:
    - Create song mashups by blending structure from two sources
    - Interpolate between musical structures
    - Create hybrid compositions

    blend_factor: 0.0 = 100% hints_a, 1.0 = 100% hints_b, 0.5 = 50/50 mix

    Accepts a scalar float for static blending, or a list of floats (e.g. from
    FeatureToFloat) for temporal blending — each value controls the blend weight
    at the corresponding time step.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hints_a": ("SEMANTIC_HINTS",),
                "hints_b": ("SEMANTIC_HINTS",),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SEMANTIC_HINTS",)
    RETURN_NAMES = ("blended_hints",)
    FUNCTION = "blend"
    CATEGORY = "audio/acestep"

    def blend(self, hints_a, hints_b, blend_factor):
        # Validate shapes match
        if hints_a.shape != hints_b.shape:
            # Try to match lengths by truncating to shorter
            min_length = min(hints_a.shape[-1], hints_b.shape[-1])
            hints_a = hints_a[..., :min_length]
            hints_b = hints_b[..., :min_length]
            logger.debug(f"[SEMANTIC_BLEND] Truncated to common length: {min_length}")

        logger.debug(f"[SEMANTIC_BLEND] Blending hints")
        logger.debug(f"[SEMANTIC_BLEND]   hints_a shape: {hints_a.shape}, stats: mean={hints_a.mean():.4f}, std={hints_a.std():.4f}")
        logger.debug(f"[SEMANTIC_BLEND]   hints_b shape: {hints_b.shape}, stats: mean={hints_b.mean():.4f}, std={hints_b.std():.4f}")

        if isinstance(blend_factor, list):
            # Temporal blending: list of per-frame weights (e.g. from FeatureToFloat)
            # Hints shape is [B, D, T] — time is the last dimension
            T = hints_a.shape[-1]
            frame_count = len(blend_factor)

            logger.debug(f"[SEMANTIC_BLEND]   temporal mode: {frame_count} weight frames -> {T} audio time steps")

            # Resample weights to match time dimension
            if frame_count == T:
                weights = blend_factor
            else:
                weights = []
                for t in range(T):
                    if T == 1:
                        frame_idx = 0
                    else:
                        frame_idx = round(t * (frame_count - 1) / (T - 1))
                    frame_idx = max(0, min(frame_idx, frame_count - 1))
                    weights.append(float(blend_factor[frame_idx]))

            # Shape [1, 1, T] to broadcast across batch and channel dims
            weight = torch.tensor(weights, dtype=hints_a.dtype, device=hints_a.device).reshape(1, 1, T)

            logger.debug(f"[SEMANTIC_BLEND]   temporal weight stats: min={weight.min():.4f}, max={weight.max():.4f}, mean={weight.mean():.4f}")

            blended = (1.0 - weight) * hints_a + weight * hints_b
        else:
            # Static blend with scalar float
            logger.debug(f"[SEMANTIC_BLEND]   blend_factor: {blend_factor} (0=A, 1=B)")
            blended = (1.0 - blend_factor) * hints_a + blend_factor * hints_b

        logger.debug(f"[SEMANTIC_BLEND]   result stats: mean={blended.mean():.4f}, std={blended.std():.4f}")

        return (blended,)


class ACEStepRepaintGuiderNode:
    """Node that creates a repaint guider for use with SamplerCustomAdvanced"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
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
        patches.apply_acestep_patches()

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
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
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
        patches.apply_acestep_patches()

        logger.debug(f"[EXTEND_GUIDER_NODE] get_guider called")
        logger.debug(f"[EXTEND_GUIDER_NODE]   extend_left_time: {extend_left_time}")
        logger.debug(f"[EXTEND_GUIDER_NODE]   extend_right_time: {extend_right_time}")

        if extend_left_time == 0 and extend_right_time == 0:
            raise ValueError("At least one of extend_left_time or extend_right_time must be > 0")

        if not validate_audio_latent(source_latents):
            raise ValueError("source_latents must be audio latents (from VAEEncodeAudio or EmptyAceStepLatentAudio)")

        # Extract latent tensor
        latent_tensor = source_latents["samples"]
        logger.debug(f"[EXTEND_GUIDER_NODE]   latent_tensor.shape: {latent_tensor.shape}")
        version = ACEStepLatentUtils.detect_version(latent_tensor)
        logger.debug(f"[EXTEND_GUIDER_NODE]   detected version: {version}")

        # Create and return the extend guider
        guider = ACEStepExtendGuider(
            model, positive, negative, cfg,
            latent_tensor, extend_left_time, extend_right_time
        )
        logger.debug(f"[EXTEND_GUIDER_NODE]   guider.is_v1_5: {guider.is_v1_5}")
        logger.debug(f"[EXTEND_GUIDER_NODE]   guider.extended_latent.shape: {guider.extended_latent.shape}")

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
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
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
        patches.apply_acestep_patches()

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
        version = ACEStepLatentUtils.detect_version(latent_tensor)
        length = latent_tensor.shape[-1]

        # Calculate duration using version-specific FPS
        duration = ACEStepLatentUtils.frame_index_to_time(length, version)

        # Create info string
        info = f"ACE Audio Latent Analysis:\n"
        info += f"Version: {version or 'unknown'}\n"
        info += f"Shape: {latent_tensor.shape}\n"
        info += f"Batch size: {latent_tensor.shape[0]}\n"

        if version == ACEStepLatentUtils.V1_5:
            info += f"Channels: {latent_tensor.shape[1]}\n"
        else:
            info += f"Channels: {latent_tensor.shape[1]}\n"
            info += f"Height: {latent_tensor.shape[2]}\n"

        info += f"Length (frames): {length}\n"
        info += f"Duration: {duration:.2f} seconds\n"
        fps = ACEStepLatentUtils.get_fps(version)
        info += f"Frame rate: {fps:.2f} frames/second"

        return (info, duration, length, latent_tensor.shape[1])

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
        version = ACEStepLatentUtils.detect_version(latent_tensor)

        # Create mask
        start_frame = ACEStepLatentUtils.time_to_frame_index(start_time, version)
        end_frame = ACEStepLatentUtils.time_to_frame_index(end_time, version)
        feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_time, version)

        mask = ACEStepLatentUtils.create_repaint_mask(latent_tensor.shape, start_frame, end_frame, version)

        if feather_frames > 0:
            feathered_mask = ACEStepLatentUtils.create_feather_mask(mask, feather_frames)
        else:
            feathered_mask = mask

        # Create visualization
        total_frames = latent_tensor.shape[-1]
        # Handle both 3D (v1.5) and 4D (v1.0) shapes
        if len(feathered_mask.shape) == 3:
            mask_summary = feathered_mask[0, 0, :].cpu().numpy()
        else:
            mask_summary = feathered_mask[0, 0, 0, :].cpu().numpy()
        
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
                # Use version-aware conversion if available
                if hasattr(guider, 'version') and guider.version == ACEStepLatentUtils.V1_5:
                    left_samples = int(guider.left_frames / 25.0 * orig_sr)  # v1.5: 25 fps
                else:
                    left_samples = int(guider.left_frames * 512 * 8 / 44100 * orig_sr)  # v1.0
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
        # Handle both v1.5 (3D) and v1.0 (4D) shapes
        if len(latent_mask.shape) == 3:
            mask_pattern = latent_mask[0, 0, :].cpu()  # v1.5: Shape: [latent_frames]
        else:
            mask_pattern = latent_mask[0, 0, 0, :].cpu()  # v1.0: Shape: [latent_frames]
        
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


# =============================================================================
# ACE-Step 1.5 Native Guider Nodes
# These use the model's native mask input via model wrapping
# =============================================================================

class ACEStep15NativeEditGuiderNode:
    """
    Unified ACE-Step 1.5 edit guider for extend and/or repaint operations.

    This guider injects chunk_masks and src_latents directly into the model's forward()
    method using model wrapping, rather than manipulating latents during sampling.

    Operations:
    - Extend: Set extend_left_seconds and/or extend_right_seconds > 0
    - Repaint: Set repaint_start_seconds and repaint_end_seconds (repaint_end > repaint_start)
    - Both: Combine extend and repaint in a single operation

    The silence_latent is automatically loaded internally (downloaded from HuggingFace
    on first use). It's a learned tensor that tells the model to generate new content.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "source_latents": ("LATENT",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            },
            "optional": {
                "extend_left_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.1}),
                "extend_right_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.1}),
                "repaint_start_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
                "repaint_end_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
                "reference_latent": ("LATENT", {"tooltip": "Optional reference audio latent for timbre conditioning. The model will generate audio with the timbre/instrument character of this reference. Does not affect task behavior (extend/repaint)."}),
            }
        }

    RETURN_TYPES = ("GUIDER", "LATENT")
    RETURN_NAMES = ("guider", "output_latent")
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, source_latents, cfg,
                   extend_left_seconds=0.0, extend_right_seconds=0.0,
                   repaint_start_seconds=-1.0, repaint_end_seconds=-1.0,
                   reference_latent=None):
        patches.apply_acestep_patches()

        source_tensor = source_latents["samples"]

        # Validate v1.5 shape: (batch, 64, length)
        if len(source_tensor.shape) != 3 or source_tensor.shape[1] != 64:
            raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_tensor.shape}. Use the v1.0 guiders for v1.0 latents.")

        # Determine which operations are enabled
        has_extend = extend_left_seconds > 0 or extend_right_seconds > 0
        has_repaint = repaint_start_seconds >= 0 and repaint_end_seconds > repaint_start_seconds

        if not has_extend and not has_repaint:
            raise ValueError("At least one operation must be enabled: set extend_left/right_seconds > 0 or set valid repaint_start/end_seconds")

        # Convert -1 values to None for the guider
        repaint_start = repaint_start_seconds if repaint_start_seconds >= 0 else None
        repaint_end = repaint_end_seconds if repaint_end_seconds >= 0 else None

        logger.debug(f"[ACE15_EDIT_NODE] Creating guider")
        logger.debug(f"[ACE15_EDIT_NODE]   source_tensor.shape: {source_tensor.shape}")
        if has_extend:
            logger.debug(f"[ACE15_EDIT_NODE]   extend: left={extend_left_seconds}s, right={extend_right_seconds}s")
        if has_repaint:
            logger.debug(f"[ACE15_EDIT_NODE]   repaint: {repaint_start_seconds}s - {repaint_end_seconds}s")

        # Extract reference latent tensor if provided
        ref_tensor = reference_latent["samples"] if reference_latent is not None else None

        # Create the unified guider (silence_latent is loaded internally)
        guider = ACEStep15NativeEditGuider(
            model, positive, negative, cfg,
            source_tensor,
            extend_left_seconds=extend_left_seconds,
            extend_right_seconds=extend_right_seconds,
            repaint_start_seconds=repaint_start,
            repaint_end_seconds=repaint_end,
            reference_latent=ref_tensor
        )

        # Return the working latent (may be extended or same as source)
        output_latent = {"samples": guider.working_latent, "type": "audio"}

        return (guider, output_latent)


class ACEStep15NativeCoverGuiderNode:
    """
    Creates an ACE-Step 1.5 cover guider for style transfer/regeneration.

    Uses semantic tokens from the source audio as structural guidance while
    generating new content with a different style/timbre.

    Semantic hints are automatically extracted if not provided. For advanced
    workflows like blending hints from multiple sources, use the Semantic
    Extractor and Semantic Hints Blend nodes.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "source_latents": ("LATENT",),
            },
            "optional": {
                "semantic_hints": ("SEMANTIC_HINTS",),
                "reference_latent": ("LATENT", {"tooltip": "Optional reference audio latent for timbre conditioning. Decouples timbre from the source audio's semantic content — the cover follows the source's structure but adopts this reference's instrument/voice character."}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg, source_latents, semantic_hints=None, reference_latent=None):
        patches.apply_acestep_patches()
        source_tensor = source_latents["samples"]

        # Validate v1.5 shape: (batch, 64, length)
        if len(source_tensor.shape) != 3 or source_tensor.shape[1] != 64:
            raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_tensor.shape}")

        # Extract reference latent tensor if provided
        ref_tensor = reference_latent["samples"] if reference_latent is not None else None

        logger.debug(f"[ACE15_COVER_NODE] Creating guider")
        logger.debug(f"[ACE15_COVER_NODE]   source_tensor.shape: {source_tensor.shape}")
        if semantic_hints is not None:
            logger.debug(f"[ACE15_COVER_NODE]   semantic_hints provided externally: {semantic_hints.shape}")
        else:
            logger.debug(f"[ACE15_COVER_NODE]   semantic_hints will be auto-extracted")
        if ref_tensor is not None:
            logger.debug(f"[ACE15_COVER_NODE]   reference_latent provided: {ref_tensor.shape}")

        guider = ACEStep15NativeCoverGuider(
            model, positive, negative, cfg, source_tensor, semantic_hints, reference_latent=ref_tensor
        )

        return (guider,)


class ACEStep15NativeExtractGuiderNode:
    """
    Creates an ACE-Step 1.5 extract guider for extracting specific tracks.

    Extracts a specific track (vocals, drums, bass, etc.) from the source audio.
    Use with the ACEStep15TaskTextEncode node with task_type="extract".

    Semantic hints are automatically extracted if not provided. For advanced
    workflows, use the Semantic Extractor node.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                         "tooltip": "Classifier-free guidance scale. Base model default is 7.0. Turbo model uses 1.0 (no CFG)."}),
                "source_latents": ("LATENT",),
                "track_name": (["vocals", "drums", "bass", "guitar", "keyboard", "strings",
                               "percussion", "synth", "fx", "brass", "woodwinds", "backing_vocals"],),
            },
            "optional": {
                "semantic_hints": ("SEMANTIC_HINTS",),
                "reference_latent": ("LATENT", {"tooltip": "Optional reference audio latent for timbre conditioning. Guides what the extracted track should sound like."}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg, source_latents, track_name, semantic_hints=None, reference_latent=None):
        patches.apply_acestep_patches()
        source_tensor = source_latents["samples"]

        # Validate v1.5 shape: (batch, 64, length)
        if len(source_tensor.shape) != 3 or source_tensor.shape[1] != 64:
            raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_tensor.shape}")

        # Extract reference latent tensor if provided
        ref_tensor = reference_latent["samples"] if reference_latent is not None else None

        logger.debug(f"[ACE15_EXTRACT_NODE] Creating guider")
        logger.debug(f"[ACE15_EXTRACT_NODE]   source_tensor.shape: {source_tensor.shape}")
        logger.debug(f"[ACE15_EXTRACT_NODE]   track_name: {track_name}")
        if semantic_hints is not None:
            logger.debug(f"[ACE15_EXTRACT_NODE]   semantic_hints provided externally: {semantic_hints.shape}")
        else:
            logger.debug(f"[ACE15_EXTRACT_NODE]   no semantic_hints (not needed for extract)")
        if ref_tensor is not None:
            logger.debug(f"[ACE15_EXTRACT_NODE]   reference_latent provided: {ref_tensor.shape}")

        guider = ACEStep15NativeExtractGuider(
            model, positive, negative, cfg, source_tensor, track_name, semantic_hints, reference_latent=ref_tensor
        )

        return (guider,)


class ACEStep15NativeLegoGuiderNode:
    """
    Creates an ACE-Step 1.5 lego guider for generating specific tracks in a region.

    Generates a specific track (vocals, drums, etc.) within a time region while
    preserving the rest. Useful for adding instruments to existing audio.
    Use with the ACEStep15TaskTextEncode node with task_type="lego".

    The silence_latent is automatically loaded internally (downloaded from HuggingFace
    on first use). It's a learned tensor that tells the model to generate new content.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                         "tooltip": "Classifier-free guidance scale. Base model default is 7.0. Turbo model uses 1.0 (no CFG)."}),
                "source_latents": ("LATENT",),
                "track_name": (["vocals", "drums", "bass", "guitar", "keyboard", "strings",
                               "percussion", "synth", "fx", "brass", "woodwinds", "backing_vocals"],),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "end_seconds": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
            },
            "optional": {
                "reference_latent": ("LATENT", {"tooltip": "Optional reference audio latent for timbre conditioning. The generated track will adopt the timbre/instrument character of this reference."}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg, source_latents,
                   track_name, start_seconds, end_seconds, reference_latent=None):
        patches.apply_acestep_patches()
        if start_seconds >= end_seconds:
            raise ValueError(f"start_seconds ({start_seconds}) must be less than end_seconds ({end_seconds})")

        source_tensor = source_latents["samples"]

        # Validate v1.5 shape: (batch, 64, length)
        if len(source_tensor.shape) != 3 or source_tensor.shape[1] != 64:
            raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_tensor.shape}")

        # Extract reference latent tensor if provided
        ref_tensor = reference_latent["samples"] if reference_latent is not None else None

        logger.debug(f"[ACE15_LEGO_NODE] Creating guider")
        logger.debug(f"[ACE15_LEGO_NODE]   source_tensor.shape: {source_tensor.shape}")
        logger.debug(f"[ACE15_LEGO_NODE]   track_name: {track_name}")
        logger.debug(f"[ACE15_LEGO_NODE]   region: {start_seconds}s - {end_seconds}s")
        if ref_tensor is not None:
            logger.debug(f"[ACE15_LEGO_NODE]   reference_latent provided: {ref_tensor.shape}")

        # Create guider (silence_latent is loaded internally)
        guider = ACEStep15NativeLegoGuider(
            model, positive, negative, cfg, source_tensor,
            track_name, start_seconds, end_seconds, reference_latent=ref_tensor
        )

        return (guider,)


class ACEStep15TaskTextEncodeNode:
    """
    Task-aware text encoder for ACE-Step 1.5.

    Encodes text with task-specific instructions:
    - text2music: Generate audio from text description
    - repaint: Regenerate a region while preserving the rest
    - cover: Style transfer/regeneration with source audio as context
    - extract: Extract a specific track (requires track_name)
    - lego: Generate a specific track in a region (requires track_name)

    The task_type determines the instruction prefix used by the model.
    For extract/lego tasks, track_name specifies which track to extract/generate.

    Returns single conditioning output. Use ConditioningZeroOut node for negative
    conditioning (standard ACE-Step workflow pattern).
    """

    # Valid keyscales from ACE-Step 1.5 reference (constants.py):
    # 7 notes × 5 accidentals ('', '#', 'b', '♯', '♭') × 2 modes = 70 combinations
    # We use ASCII only ('#', 'b') plus 'Db/Eb/Gb/Ab/Bb' enharmonic spellings = 56 unique
    KEYSCALE_NOTES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    KEYSCALE_ACCIDENTALS = ['', '#', 'b']
    KEYSCALE_MODES = ['major', 'minor']

    VALID_KEYSCALES = []
    for note in KEYSCALE_NOTES:
        for acc in KEYSCALE_ACCIDENTALS:
            for mode in KEYSCALE_MODES:
                VALID_KEYSCALES.append(f"{note}{acc} {mode}")

    # Valid time signatures from reference
    VALID_TIME_SIGNATURES = ["2", "3", "4", "6"]

    # Valid languages from reference
    VALID_LANGUAGES = [
        'en', 'zh', 'ja', 'ko', 'es', 'fr', 'de', 'it', 'pt', 'ru',
        'ar', 'hi', 'vi', 'th', 'id', 'ms', 'tl', 'nl', 'pl', 'tr',
        'sv', 'da', 'no', 'fi', 'cs', 'sk', 'hu', 'ro', 'bg', 'hr',
        'sr', 'uk', 'el', 'he', 'fa', 'bn', 'ta', 'te', 'pa', 'ur',
        'ne', 'sw', 'ht', 'is', 'lt', 'la', 'az', 'ca', 'sa', 'yue',
        'unknown'
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A melodic electronic track with soft synths"}),
                "task_type": (["text2music", "repaint", "cover", "extract", "lego"],
                             {"tooltip": "text2music/repaint use LM audio code generation (cfg_scale, temperature, top_p, top_k apply). cover/extract/lego use precomputed semantic hints from source audio instead."}),
            },
            "optional": {
                "track_name": (["", "vocals", "drums", "bass", "guitar", "keyboard", "strings",
                               "percussion", "synth", "fx", "brass", "woodwinds", "backing_vocals"],
                              {"default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "bpm": ("INT", {"default": 120, "min": 10, "max": 300}),
                "duration": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 2000.0, "step": 0.1}),
                # Use COMBO type to accept connections from AudioInfo detected_key output
                "keyscale": ("COMBO", {"default": "C major", "options": s.VALID_KEYSCALES}),
                "timesignature": ("COMBO", {"default": "4", "options": s.VALID_TIME_SIGNATURES}),
                "language": ("COMBO", {"default": "en", "options": s.VALID_LANGUAGES}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1,
                              "tooltip": "Controls how closely the generated audio follows your text prompt. Higher values produce output that matches your description more literally, lower values allow more freedom. No effect on cover/extract/lego tasks, which use semantic hints from source audio instead of generating new audio codes."}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01,
                                 "tooltip": "Controls randomness and creativity in the generated audio. Lower values (0.7-0.85) produce more consistent, predictable results. Higher values (0.9-1.1) produce more varied, surprising output. No effect on cover/extract/lego tasks, which use semantic hints from source audio instead of generating new audio codes."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                           "tooltip": "Limits how many possible audio choices are considered at each step. Lower values (e.g. 0.8) produce safer, more predictable output. Higher values allow more diversity. 1.0 disables this filter. No effect on cover/extract/lego tasks, which use semantic hints from source audio instead of generating new audio codes."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100,
                           "tooltip": "Restricts each generation step to only the top K most likely choices. 0 disables this filter. Lower values (e.g. 40) reduce unlikely outputs while keeping variety. No effect on cover/extract/lego tasks, which use semantic hints from source audio instead of generating new audio codes."}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                           "tooltip": "Minimum probability threshold for token sampling. Filters out tokens with probability below min_p × max_probability. 0.0 disables this filter. No effect on cover/extract/lego tasks."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, task_type, track_name="", lyrics="", bpm=120,
               duration=60.0, keyscale="C major", timesignature="4", language="en", seed=0,
               cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0):
        patches.apply_acestep_patches()

        # Validate track_name for extract/lego tasks
        if task_type in ["extract", "lego"] and not track_name:
            logger.debug(f"[ACE15_TEXT_ENCODE] Warning: track_name not specified for {task_type} task, using default instruction")

        # Convert timesignature from string to int
        timesig_int = int(timesignature)

        # Build kwargs for tokenizer
        kwargs = {
            "lyrics": lyrics,
            "bpm": bpm,
            "duration": duration,
            "keyscale": keyscale,
            "timesignature": timesig_int,
            "language": language,
            "seed": seed,
            "task_type": task_type,
            "track_name": track_name if track_name else None,
            "cfg_scale": cfg_scale,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
        }

        logger.debug(f"[ACE15_TEXT_ENCODE] Encoding with task_type={task_type}, track_name={track_name or 'N/A'}")
        logger.debug(f"[ACE15_TEXT_ENCODE] LM params: cfg_scale={cfg_scale}, temperature={temperature}, top_p={top_p}, top_k={top_k}")

        # Import to get task instruction for verification
        from .patches import get_task_instruction, is_patched
        instruction = get_task_instruction(task_type, track_name if track_name else None)
        logger.debug(f"[ACE15_TEXT_ENCODE]   patches applied: {is_patched()}")
        logger.debug(f"[ACE15_TEXT_ENCODE]   task instruction: {instruction}")

        # Use the patched tokenize_with_weights which accepts task_type
        # encode_from_tokens_scheduled returns conditioning directly (list format)
        tokens = clip.tokenize(text, **kwargs)

        # Verify the tokens contain task_type info (patched tokenizer should include this)
        if isinstance(tokens, dict):
            if "task_type" in tokens:
                logger.debug(f"[ACE15_TEXT_ENCODE]   tokens contain task_type: {tokens.get('task_type')}")
            else:
                logger.debug(f"[ACE15_TEXT_ENCODE]   WARNING: tokens dict does not contain task_type - patch may not be applied!")
                logger.debug(f"[ACE15_TEXT_ENCODE]   tokens keys: {list(tokens.keys())}")

        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # For negative conditioning, use ComfyUI's ConditioningZeroOut node
        # This matches the standard ACE-Step workflow pattern

        return (conditioning,)


class ACEStep15KeystoneConfig:
    """Configuration node for ACE-Step 1.5 keystone channel gains.

    Keystone channels are the 5 individual latent channels with outsized impact on
    audio generation. Each controls a distinct timbral quality. Connect the output
    to the Generation Steering or Latent Channel EQ node's keystone_config input.

    Redesigned based on Phase 1+2 multi-seed statistical validation.
    ch14 and ch23 dropped (not statistically significant across seeds).
    ch2 added (significant weight control). Inverse channels (ch56, ch13)
    now have negative sensitivity so the UI works intuitively.
    """

    DESCRIPTION = (
        "[Experimental] Configures gains for 5 keystone latent channels with outsized individual impact.\n\n"
        "- presence (ch19): +8% RMS, +3% centroid. Late-stage dominant. "
        "Most impactful single channel for timbral character.\n"
        "- spectral_tilt (ch29): -13% centroid, +8% RMS. Late-stage. "
        "Shifts the entire spectrum brighter or darker.\n"
        "- energy (ch56): +13% RMS, +7% centroid, +3% onset. INVERSE: "
        "boosting this slider attenuates ch56, which adds energy/shimmer.\n"
        "- brilliance (ch13): +18.5% centroid. INVERSE: boosting adds brilliance. "
        "Late-stage dominant.\n"
        "- weight (ch2): -4.3% RMS, spectrally neutral. Late-stage bass weight control.\n\n"
        "Connect output to Generation Steering or Latent Channel EQ's keystone_config input."
    )

    # Sensitivity factors: internal_gain = 1.0 + factor * (user_gain - 1.0)
    # Negative = INVERTED (user "UP" → internal gain < 1 → attenuates channel → more of named quality)
    KEYSTONE_SENSITIVITY = {
        "presence": 1.8,        # ch19: partial compensation, push harder
        "spectral_tilt": 1.2,   # ch29: partial compensation
        "energy": -1.0,         # ch56: INVERTED — attenuating adds energy
        "brilliance": -0.7,     # ch13: INVERTED — attenuating adds brilliance
        "weight": 1.0,          # ch2: near-linear
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "presence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Channel 19: Presence/definition. Most impactful channel. +8% RMS, +3% centroid at gs=1."}),
                "spectral_tilt": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Channel 29: Spectral brightness. -13% centroid, +8% RMS. Reducing shifts darker."}),
                "energy": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Channel 56: Energy/shimmer. Inverse: boosting adds +13% RMS, +7% centroid, +3% onset."}),
                "brilliance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Channel 13: Brilliance. Inverse: boosting adds +18.5% centroid. Late-stage dominant."}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Channel 2: Bass weight. -4.3% RMS, spectrally neutral. Late-stage."}),
            },
        }

    RETURN_TYPES = ("KEYSTONE_CONFIG",)
    RETURN_NAMES = ("keystone_config",)
    FUNCTION = "build_config"
    CATEGORY = "audio/acestep"

    # Channel index mapping for each keystone parameter
    KEYSTONE_CHANNELS = {
        "presence": 19,
        "spectral_tilt": 29,
        "energy": 56,
        "brilliance": 13,
        "weight": 2,
    }

    def build_config(self, presence, spectral_tilt, energy, brilliance, weight):
        config = {}
        params = {
            "presence": presence,
            "spectral_tilt": spectral_tilt,
            "energy": energy,
            "brilliance": brilliance,
            "weight": weight,
        }
        for name, value in params.items():
            sensitivity = self.KEYSTONE_SENSITIVITY[name]
            if isinstance(value, list):
                # Apply sensitivity per-element for temporal scheduling
                normalized = [1.0 + sensitivity * (v - 1.0) for v in value]
                config[self.KEYSTONE_CHANNELS[name]] = normalized
            else:
                if abs(value - 1.0) < 1e-6:
                    continue
                config[self.KEYSTONE_CHANNELS[name]] = 1.0 + sensitivity * (value - 1.0)
        return (config,)


class _LatentChannelBandMixin:
    """Shared infrastructure for 6-band latent channel nodes (Generation Steering + EQ).

    Contains band definitions, sensitivity factors, gain tensor building, and temporal utilities.
    Not a ComfyUI node — just a mixin for code reuse.
    """

    # 6 semantic bands mapping band index to channel indices
    # Redesigned based on Phase 1+2 multi-seed findings (10 seeds × 6 genres + 5 seeds × 4 genres)
    BAND_CHANNELS = {
        0: list(range(0, 8)) + list(range(32, 40)),    # bass (G0+G4)
        1: list(range(8, 16)),                           # brightness (G1 alone)
        2: list(range(40, 48)) + list(range(48, 56)),    # body (G5+G6)
        3: list(range(16, 24)),                          # texture (G2)
        4: list(range(24, 32)),                          # tilt (G3)
        5: list(range(56, 64)),                          # air (G7)
    }

    # Per-band sensitivity factors: internal_gain = 1.0 + factor * (user_gain - 1.0)
    # Derived from Phase 2 guidance compensation ratios across multi-seed grid.
    # Negative = INVERTED (attenuating increases the named quality).
    BAND_SENSITIVITY = {
        0: 0.9,     # bass — near-linear (G0=SIMILAR 1.10, G4=SIMILAR 1.16)
        1: -0.7,    # brightness — INVERTED, amplifies (G1=EMERGENT 1.30)
        2: 1.0,     # body — direct, G5=SIMILAR 0.83, G6=COMPENSATE 0.31, blended ~1.0
        3: 0.5,     # texture — direct, conservative (G2=EMERGENT 0.18, very unpredictable)
        4: 1.5,     # tilt — direct, compensated (G3=SIMILAR 0.55)
        5: -1.0,    # air — INVERTED, near-linear (G7=SIMILAR 0.87)
    }

    # Shared band input definitions for INPUT_TYPES
    @staticmethod
    def _band_inputs():
        return {
            "bass_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Channels 0-7, 32-39: Low-frequency energy and overall volume."}),
            "brightness_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Channels 8-15: Spectral brightness. Strongest spectral control."}),
            "body_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Channels 40-55: Broadband fullness."}),
            "texture_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Channels 16-23: Mid-range timbral character. Effects vary by genre."}),
            "tilt_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Channels 24-31: Spectral balance. Higher = brighter spectrum."}),
            "air_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Channels 56-63: High-frequency shimmer, energy, and rhythmic activity."}),
        }

    @staticmethod
    def _effect_range_inputs():
        return {
            "effect_start_pct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Denoising progress to start applying effect (0=from start)"}),
            "effect_end_pct": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Denoising progress to stop applying effect (1=until end)"}),
        }

    @staticmethod
    def _gain_to_temporal(gain, T, device, dtype):
        """Convert a gain value (scalar or list) to a tensor of shape [T]."""
        if isinstance(gain, list):
            frame_count = len(gain)
            if frame_count == T:
                return torch.tensor(gain, dtype=dtype, device=device)
            weights = []
            for t in range(T):
                if T == 1:
                    frame_idx = 0
                else:
                    frame_idx = round(t * (frame_count - 1) / (T - 1))
                frame_idx = max(0, min(frame_idx, frame_count - 1))
                weights.append(float(gain[frame_idx]))
            return torch.tensor(weights, dtype=dtype, device=device)
        else:
            return torch.full((T,), float(gain), dtype=dtype, device=device)

    @staticmethod
    def _build_gain_tensor(all_gains, n_channels, T, device, dtype, temporal_mask=None, keystone_config=None):
        """Build gain tensor [1, n_channels, T] with optional temporal mask blending and keystone overlay."""
        gain_tensor = torch.ones(1, n_channels, T, device=device, dtype=dtype)

        # Assign band gains with per-band sensitivity normalization
        for band_idx, channels in _LatentChannelBandMixin.BAND_CHANNELS.items():
            band_gain = _LatentChannelBandMixin._gain_to_temporal(
                all_gains[band_idx], T, device, dtype
            )
            # Apply sensitivity: internal = 1.0 + sensitivity * (user - 1.0)
            sensitivity = _LatentChannelBandMixin.BAND_SENSITIVITY[band_idx]
            band_gain = 1.0 + sensitivity * (band_gain - 1.0)
            for ch in channels:
                if ch < n_channels:
                    gain_tensor[:, ch, :] = band_gain

        # Layer keystone config on top (multiply with band gain)
        if keystone_config:
            for ch_idx, ks_gain in keystone_config.items():
                if ch_idx < n_channels:
                    ks_temporal = _LatentChannelBandMixin._gain_to_temporal(
                        ks_gain, T, device, dtype
                    )
                    gain_tensor[:, ch_idx, :] *= ks_temporal

        if temporal_mask is not None:
            mask = temporal_mask.to(device=device, dtype=dtype)
            while mask.ndim < 3:
                mask = mask.unsqueeze(0)
            if mask.shape[-1] != T:
                mask = torch.nn.functional.interpolate(
                    mask, size=T, mode='linear', align_corners=False
                )
            # Broadcast: where mask=1 use EQ gains, where mask=0 use neutral
            gain_tensor = 1.0 + mask * (gain_tensor - 1.0)

        return gain_tensor

    @staticmethod
    def _check_progress(sigma, model_options, effect_start_pct, effect_end_pct):
        """Return True if current denoising progress is outside the effect range."""
        sigmas = model_options.get("transformer_options", {}).get("sample_sigmas", None)
        if sigmas is not None:
            sigma_max = float(sigmas[0])
            sigma_min = float(sigmas[-1])
            denom = sigma_max - sigma_min
            if denom > 1e-7:
                progress = 1.0 - (float(sigma) - sigma_min) / denom
                if progress < effect_start_pct or progress > effect_end_pct:
                    return True
        return False

    @staticmethod
    def _collect_gains(bass_gain, brightness_gain, body_gain, texture_gain, tilt_gain, air_gain):
        return [bass_gain, brightness_gain, body_gain, texture_gain, tilt_gain, air_gain]

    @staticmethod
    def _gains_are_neutral(all_gains):
        return all(not isinstance(g, list) and abs(g - 1.0) < 1e-6 for g in all_gains)


class ACEStep15GenerationSteering(_LatentChannelBandMixin):
    """Steers audio generation via latent channel guidance during diffusion.

    The primary creative tool for shaping ACE-Step 1.5 output. Runs the model twice per step —
    once with normal input, once with channel-scaled input — then steers generation along the
    difference. This changes what the model produces, not just how it sounds.
    """

    DESCRIPTION = (
        "[Experimental] Steers audio generation by running the model twice per diffusion step — "
        "once normally, once with channel-scaled input — and steering along the difference.\n\n"
        "This is NOT post-processing EQ. It changes what the model generates: timbral character, "
        "spectral balance, rhythmic activity, and harmonic content. The 64 channels are grouped "
        "into 6 semantically meaningful bands based on Phase 1+2 multi-seed validation. "
        "All sliders range 0-2 with 1.0 as neutral. Brightness and air bands are INVERTED "
        "(boosting the slider attenuates those channels, which increases brightness/air).\n\n"
        "BANDS:\n"
        "- bass (ch 0-7, 32-39): Low-frequency energy and overall volume.\n"
        "- brightness (ch 8-15): Spectral brightness. Strongest spectral control. INVERTED.\n"
        "- body (ch 40-55): Broadband fullness.\n"
        "- texture (ch 16-23): Mid-range timbral character. Effects vary by genre.\n"
        "- tilt (ch 24-31): Spectral balance. Higher = brighter spectrum.\n"
        "- air (ch 56-63): High-frequency shimmer, energy, and rhythmic activity. INVERTED.\n\n"
        "OPTIONAL: Connect a Keystone Config node to fine-tune the 5 most impactful individual channels.\n\n"
        "guidance_scale controls strength. Positive steers toward the emphasis, negative steers away. "
        "Each step runs an extra model forward pass."
    )

    @classmethod
    def INPUT_TYPES(s):
        required = {"model": ("MODEL",)}
        required["guidance_scale"] = ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1,
            "tooltip": "Guidance strength. 0=off, positive=steer toward emphasis, negative=steer away. Accepts a list of floats for temporal scheduling."})
        required.update(s._band_inputs())
        required.update(s._effect_range_inputs())
        return {
            "required": required,
            "optional": {
                "temporal_mask": ("MASK", {"tooltip": "Blends gains toward neutral (1.0) where mask is 0."}),
                "keystone_config": ("KEYSTONE_CONFIG", {"tooltip": "Optional keystone channel config. Multiplies on top of band gains."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "audio/acestep"

    def apply(self, model, guidance_scale,
              bass_gain, brightness_gain, body_gain, texture_gain, tilt_gain, air_gain,
              effect_start_pct, effect_end_pct, temporal_mask=None, keystone_config=None):
        m = model.clone()

        all_gains = self._collect_gains(bass_gain, brightness_gain, body_gain, texture_gain, tilt_gain, air_gain)
        gains_neutral = self._gains_are_neutral(all_gains)
        has_keystone = keystone_config is not None and len(keystone_config) > 0

        gs_is_zero = not isinstance(guidance_scale, list) and abs(guidance_scale) < 1e-6
        if (gains_neutral and not has_keystone) or gs_is_zero:
            return (m,)

        def wrapper_fn(apply_model_fn, args):
            x_in = args["input"]
            t_in = args["timestep"]
            c_in = args["c"]
            model_options = {"transformer_options": c_in.get("transformer_options", {})}
            skip = self._check_progress(t_in, model_options, effect_start_pct, effect_end_pct)

            # Normal forward pass
            output = apply_model_fn(x_in, t_in, **c_in)

            if skip:
                return output

            gain = self._build_gain_tensor(all_gains, x_in.shape[1], x_in.shape[-1],
                                           x_in.device, x_in.dtype, temporal_mask, keystone_config)

            # Guidance: run model again with channel-scaled input, steer via difference
            x_eq = x_in * gain
            eq_output = apply_model_fn(x_eq, t_in, **c_in)
            guidance_direction = eq_output - output

            # Broadcast guidance_scale across time dimension: [1, 1, T]
            gs = self._gain_to_temporal(guidance_scale, x_in.shape[-1], x_in.device, x_in.dtype)
            gs = gs.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            output = output + gs * guidance_direction

            return output

        m.model_options["model_function_wrapper"] = wrapper_fn
        return (m,)


class ACEStep15LatentChannelEQ(_LatentChannelBandMixin):
    """Multiplicative latent-space channel EQ for ACE-Step 1.5.

    Linearly scales model output at various pipeline points. This is closer to traditional
    post-processing EQ — it changes how the output sounds, but not what the model generates.
    For creative steering of the generation process itself, use Generation Steering instead.
    """

    DESCRIPTION = (
        "[Experimental] Multiplicative 6-band EQ for ACE-Step 1.5 latent channels.\n\n"
        "Scales model output at a chosen pipeline point. This is equivalent to post-processing "
        "EQ — it shapes the frequency balance of the output but does not change what the model "
        "generates. For creative control over the generation process itself, use the Generation "
        "Steering node instead.\n\n"
        "BANDS:\n"
        "- bass (ch 0-7, 32-39): Low-frequency energy and overall volume.\n"
        "- brightness (ch 8-15): Spectral brightness. Strongest spectral control. INVERTED.\n"
        "- body (ch 40-55): Broadband fullness.\n"
        "- texture (ch 16-23): Mid-range timbral character. Effects vary by genre.\n"
        "- tilt (ch 24-31): Spectral balance. Higher = brighter spectrum.\n"
        "- air (ch 56-63): High-frequency shimmer, energy, and rhythmic activity. INVERTED.\n\n"
        "MODES:\n"
        "- post_cfg: scales final denoised output (most common)\n"
        "- pre_cfg_cond_only: scales conditional prediction only\n"
        "- pre_cfg_both: scales cond and uncond predictions\n"
        "- model_wrapper: scales each forward pass output"
    )

    HOOK_MODES = ["post_cfg", "pre_cfg_cond_only", "pre_cfg_both", "model_wrapper"]

    @classmethod
    def INPUT_TYPES(s):
        required = {"model": ("MODEL",)}
        required["hook_mode"] = (s.HOOK_MODES, {"default": "post_cfg",
            "tooltip": "Pipeline point to apply multiplicative scaling."})
        required.update(s._band_inputs())
        required.update(s._effect_range_inputs())
        return {
            "required": required,
            "optional": {
                "temporal_mask": ("MASK", {"tooltip": "Blends gains toward neutral (1.0) where mask is 0."}),
                "keystone_config": ("KEYSTONE_CONFIG", {"tooltip": "Optional keystone channel config. Multiplies on top of band gains."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "audio/acestep"

    def apply(self, model, hook_mode,
              bass_gain, brightness_gain, body_gain, texture_gain, tilt_gain, air_gain,
              effect_start_pct, effect_end_pct, temporal_mask=None, keystone_config=None):
        m = model.clone()

        all_gains = self._collect_gains(bass_gain, brightness_gain, body_gain, texture_gain, tilt_gain, air_gain)
        gains_neutral = self._gains_are_neutral(all_gains)
        has_keystone = keystone_config is not None and len(keystone_config) > 0

        if gains_neutral and temporal_mask is None and not has_keystone:
            return (m,)

        if hook_mode == "model_wrapper":
            def wrapper_fn(apply_model_fn, args):
                x_in = args["input"]
                t_in = args["timestep"]
                c_in = args["c"]
                model_options = {"transformer_options": c_in.get("transformer_options", {})}
                output = apply_model_fn(x_in, t_in, **c_in)
                if self._check_progress(t_in, model_options, effect_start_pct, effect_end_pct):
                    return output
                gain = self._build_gain_tensor(all_gains, x_in.shape[1], x_in.shape[-1],
                                               x_in.device, x_in.dtype, temporal_mask, keystone_config)
                return output * gain

            m.model_options["model_function_wrapper"] = wrapper_fn

        elif hook_mode == "post_cfg":
            def post_cfg_fn(args):
                denoised = args["denoised"]
                if self._check_progress(args["sigma"], args["model_options"], effect_start_pct, effect_end_pct):
                    return denoised
                gain = self._build_gain_tensor(all_gains, denoised.shape[1], denoised.shape[-1],
                                               denoised.device, denoised.dtype, temporal_mask, keystone_config)
                return denoised * gain

            existing = m.model_options.get("sampler_post_cfg_function", [])
            m.model_options["sampler_post_cfg_function"] = existing + [post_cfg_fn]

        elif hook_mode in ("pre_cfg_cond_only", "pre_cfg_both"):
            cond_only = hook_mode == "pre_cfg_cond_only"

            def pre_cfg_fn(args):
                conds_out = args["conds_out"]
                if self._check_progress(args["sigma"], args["model_options"], effect_start_pct, effect_end_pct):
                    return conds_out

                result = list(conds_out)
                for idx in range(len(result)):
                    if cond_only and idx != 0:
                        continue
                    pred = result[idx]
                    gain = self._build_gain_tensor(all_gains, pred.shape[1], pred.shape[-1],
                                                   pred.device, pred.dtype, temporal_mask, keystone_config)
                    result[idx] = pred * gain
                return result

            existing = m.model_options.get("sampler_pre_cfg_function", [])
            m.model_options["sampler_pre_cfg_function"] = existing + [pre_cfg_fn]

        return (m,)


class ACEStep15MusicalControls(_LatentChannelBandMixin):
    """High-level musical controls for ACE-Step 1.5, derived from Phase 3 research.

    Translates intuitive musical properties into the specific band+keystone recipes
    that Phase 3 proved reliably control those properties across seeds and genres.
    Wraps Generation Steering internally — connect MODEL in, get MODEL out.
    """

    DESCRIPTION = (
        "[Experimental — NEEDS PHASE 3 RE-VALIDATION] Musical-level steering for ACE-Step 1.5.\n\n"
        "Each slider controls a musically meaningful property. Band/keystone definitions were "
        "redesigned in Phase 1+2 validation. These recipes reference the NEW band/keystone names "
        "but the specific gain values need re-tuning after Phase 3 experiments with the new "
        "groupings. Use with caution — results may not match descriptions until re-validated.\n\n"
        "- rhythmic_density: Sparse (0) ↔ Dense (2). Controls how many rhythmic events occur.\n"
        "- rhythmic_regularity: Loose/syncopated (0) ↔ Locked/metronomic (2).\n"
        "- harmonic_complexity: Simple/tonal (0) ↔ Rich/chromatic (2).\n"
        "- instrument_independence: Unified hits (0) ↔ Independent layers (2).\n"
        "- tonality: Tonal/clean (0) ↔ Noisy/breathy (2).\n"
        "- dynamics: Compressed/flat (0) ↔ Dynamic/breathing (2).\n\n"
        "These map to specific combinations of latent channel bands and keystone channels "
        "identified through systematic experimentation."
    )

    # Recipe definitions: each musical control maps to band gains and keystone gains.
    # Values represent the gain at slider=0 and slider=2 (slider=1 is neutral).
    # Format: {param_name: (value_at_0, value_at_2)}
    # The slider linearly interpolates between these.
    # WARNING: These recipes were ported from old band/keystone names to new Phase 1+2
    # names but have NOT been re-validated with Phase 3 experiments. The specific gain
    # values need re-tuning after Phase 3 runs with the new groupings.
    RECIPES = {
        "rhythmic_density": {
            # NEEDS PHASE 3 RE-VALIDATION
            # Old recipe used all 6 keystones. New config has 5 keystones.
            "keystones": {
                "presence": (1.3, 0.7),
                "spectral_tilt": (1.3, 0.7),
                "energy": (1.3, 0.7),
                "brilliance": (1.3, 0.7),
                "weight": (1.3, 0.7),
            },
            "bands": {},
        },
        "rhythmic_regularity": {
            # NEEDS PHASE 3 RE-VALIDATION
            # Old: texture↑+balance↓. New: texture↑+tilt↓ (balance→tilt)
            "bands": {
                "texture": (0.5, 1.5),
                "tilt": (1.5, 0.5),
            },
            "keystones": {},
        },
        "harmonic_complexity": {
            # NEEDS PHASE 3 RE-VALIDATION
            # Old: foundation+weight bands. New: bass+body (foundation→bass, weight→body)
            "bands": {
                "bass": (1.3, 0.7),
                "body": (1.5, 0.5),
            },
            "keystones": {},
        },
        "instrument_independence": {
            # NEEDS PHASE 3 RE-VALIDATION
            # Old: balance band + body keystone. New: tilt band + weight keystone
            "bands": {
                "tilt": (0.5, 1.25),
            },
            "keystones": {
                "weight": (1.3, 0.7),
            },
        },
        "tonality": {
            # NEEDS PHASE 3 RE-VALIDATION
            # Old: weight+foundation bands + spectral_tilt keystone.
            # New: body+bass bands + spectral_tilt keystone
            "bands": {
                "body": (1.3, 0.7),
                "bass": (1.3, 0.7),
            },
            "keystones": {
                "spectral_tilt": (0.7, 1.3),
            },
        },
        "dynamics": {
            # NEEDS PHASE 3 RE-VALIDATION
            # Old: body band + air band + body keystone.
            # New: body band + air band + weight keystone
            "bands": {
                "body": (1.3, 0.7),
                "air": (0.7, 1.3),
            },
            "keystones": {
                "weight": (0.7, 1.3),
            },
        },
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Guidance strength. Accepts list of floats for temporal scheduling."}),
                "rhythmic_density": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sparse (0) ↔ Dense (2). Controls number of rhythmic events."}),
                "rhythmic_regularity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Loose/syncopated (0) ↔ Locked/metronomic (2). Emergent groove lock at high values."}),
                "harmonic_complexity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Simple/tonal (0) ↔ Rich/chromatic (2). Controls chord movement and pitch diversity."}),
                "instrument_independence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Unified hits (0) ↔ Independent layers (2). Controls whether instruments trigger together or independently."}),
                "tonality": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Tonal/clean (0) ↔ Noisy/breathy (2). Shifts between pitched and noise-like character."}),
                "dynamics": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Compressed/flat (0) ↔ Dynamic/breathing (2). Controls loudness variation over time."}),
                "effect_start_pct": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Denoising progress to start applying effect (0=from start)"}),
                "effect_end_pct": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Denoising progress to stop applying effect (1=until end)"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "audio/acestep"

    @staticmethod
    def _interpolate_recipe_value(slider, val_at_0, val_at_2):
        """Interpolate a recipe parameter based on slider position.

        slider=0 → val_at_0, slider=1 → 1.0 (neutral), slider=2 → val_at_2.
        Handles both scalar and list inputs.
        """
        if isinstance(slider, list):
            return [ACEStep15MusicalControls._interpolate_recipe_value(s, val_at_0, val_at_2) for s in slider]
        if slider <= 1.0:
            # Interpolate between val_at_0 and 1.0
            t = slider  # 0→0, 1→1
            return val_at_0 + t * (1.0 - val_at_0)
        else:
            # Interpolate between 1.0 and val_at_2
            t = slider - 1.0  # 0→0, 1→1
            return 1.0 + t * (val_at_2 - 1.0)

    def _compose_gains(self, **musical_params):
        """Compose all musical sliders into band gains and keystone gains."""
        # Accumulate deviations from neutral for each band and keystone
        band_accum = {b: [] for b in ["bass", "brightness", "body", "texture", "tilt", "air"]}
        ks_accum = {k: [] for k in ["presence", "spectral_tilt", "energy", "brilliance", "weight"]}

        for control_name, slider_value in musical_params.items():
            recipe = self.RECIPES.get(control_name)
            if not recipe:
                continue

            # Skip neutral sliders
            if not isinstance(slider_value, list) and abs(slider_value - 1.0) < 1e-6:
                continue

            for band_name, (v0, v2) in recipe.get("bands", {}).items():
                val = self._interpolate_recipe_value(slider_value, v0, v2)
                band_accum[band_name].append(val)

            for ks_name, (v0, v2) in recipe.get("keystones", {}).items():
                val = self._interpolate_recipe_value(slider_value, v0, v2)
                ks_accum[ks_name].append(val)

        # Combine multiple contributions: multiply deviations from 1.0
        # final = 1.0 + sum(deviation_i) where deviation_i = (val_i - 1.0)
        band_gains = {}
        band_name_to_idx = {"bass": 0, "brightness": 1, "body": 2, "texture": 3, "tilt": 4, "air": 5}
        for band_name, contributions in band_accum.items():
            if not contributions:
                band_gains[band_name] = 1.0
            elif len(contributions) == 1:
                band_gains[band_name] = contributions[0]
            else:
                # Sum deviations from neutral
                band_gains[band_name] = self._sum_deviations(contributions)

        ks_gains = {}
        for ks_name, contributions in ks_accum.items():
            if not contributions:
                continue
            elif len(contributions) == 1:
                val = contributions[0]
                if isinstance(val, list) or abs(val - 1.0) >= 1e-6:
                    ks_gains[ks_name] = val
            else:
                combined = self._sum_deviations(contributions)
                if isinstance(combined, list) or abs(combined - 1.0) >= 1e-6:
                    ks_gains[ks_name] = combined

        return band_gains, ks_gains

    @staticmethod
    def _sum_deviations(contributions):
        """Sum deviations from 1.0 across multiple contributions. Handles lists."""
        if any(isinstance(c, list) for c in contributions):
            # Find max length
            max_len = max(len(c) if isinstance(c, list) else 1 for c in contributions)
            result = []
            for i in range(max_len):
                total_dev = 0.0
                for c in contributions:
                    if isinstance(c, list):
                        v = c[min(i, len(c) - 1)]
                    else:
                        v = c
                    total_dev += (v - 1.0)
                result.append(1.0 + total_dev)
            return result
        else:
            total_dev = sum(c - 1.0 for c in contributions)
            return 1.0 + total_dev

    def apply(self, model, guidance_scale,
              rhythmic_density, rhythmic_regularity, harmonic_complexity,
              instrument_independence, tonality, dynamics,
              effect_start_pct, effect_end_pct):

        musical_params = {
            "rhythmic_density": rhythmic_density,
            "rhythmic_regularity": rhythmic_regularity,
            "harmonic_complexity": harmonic_complexity,
            "instrument_independence": instrument_independence,
            "tonality": tonality,
            "dynamics": dynamics,
        }

        # Check if everything is neutral
        all_neutral = all(
            not isinstance(v, list) and abs(v - 1.0) < 1e-6
            for v in musical_params.values()
        )
        gs_is_zero = not isinstance(guidance_scale, list) and abs(guidance_scale) < 1e-6
        if all_neutral or gs_is_zero:
            return (model.clone(),)

        # Decompose musical sliders into band + keystone gains
        band_gains, ks_gains = self._compose_gains(**musical_params)

        # Build keystone config (apply sensitivity normalization like KeystoneConfig node does)
        keystone_config = None
        if ks_gains:
            keystone_config = {}
            for name, value in ks_gains.items():
                sensitivity = ACEStep15KeystoneConfig.KEYSTONE_SENSITIVITY[name]
                ch_idx = ACEStep15KeystoneConfig.KEYSTONE_CHANNELS[name]
                if isinstance(value, list):
                    keystone_config[ch_idx] = [1.0 + sensitivity * (v - 1.0) for v in value]
                else:
                    keystone_config[ch_idx] = 1.0 + sensitivity * (value - 1.0)

        # Collect band gains in order
        band_order = ["bass", "brightness", "body", "texture", "tilt", "air"]
        all_gains = [band_gains.get(b, 1.0) for b in band_order]

        # Apply via Generation Steering's guidance mechanism
        m = model.clone()

        def wrapper_fn(apply_model_fn, args):
            x_in = args["input"]
            t_in = args["timestep"]
            c_in = args["c"]
            model_options = {"transformer_options": c_in.get("transformer_options", {})}
            skip = self._check_progress(t_in, model_options, effect_start_pct, effect_end_pct)

            output = apply_model_fn(x_in, t_in, **c_in)

            if skip:
                return output

            gain = self._build_gain_tensor(all_gains, x_in.shape[1], x_in.shape[-1],
                                           x_in.device, x_in.dtype, None, keystone_config)

            x_eq = x_in * gain
            eq_output = apply_model_fn(x_eq, t_in, **c_in)
            guidance_direction = eq_output - output

            gs = self._gain_to_temporal(guidance_scale, x_in.shape[-1], x_in.device, x_in.dtype)
            gs = gs.unsqueeze(0).unsqueeze(0)
            output = output + gs * guidance_direction

            return output

        m.model_options["model_function_wrapper"] = wrapper_fn
        return (m,)


class ModelSamplingACEStep:
    """Override model sampling for ACE-Step 1.5.

    ACE-Step uses ModelSamplingDiscreteFlow with multiplier=1.0.
    The shift parameter controls the sigma schedule:
    - shift=3.0 for turbo model (default in supported_models)
    - shift=1.0 for base model (linear spacing)

    Place this node BEFORE the scheduler so sigmas are computed with the correct shift.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "audio/acestep"

    def patch(self, model, shift):
        m = model.clone()

        sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=1.0)
        m.add_object_patch("model_sampling", model_sampling)
        return (m,)


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # ACE-Step 1.0 guiders (latent-level ODE blending)
    "ACEStepRepaintGuider": ACEStepRepaintGuiderNode,
    "ACEStepExtendGuider": ACEStepExtendGuiderNode,
    "ACEStepHybridGuider": ACEStepHybridGuiderNode,
    # Utility nodes
    "ACEStepAnalyzeLatent": ACEStepAnalyzeLatentNode,
    "ACEStepTimeRange": ACEStepTimeRangeNode,
    "ACEStepMaskVisualizer": ACEStepMaskVisualizerNode,
    "ACEStepAudioPostProcessor": ACEStepAudioPostProcessor,
    # ACE-Step 1.5 semantic extraction and blending
    "ACEStep15SemanticExtractor": ACEStep15SemanticExtractor,
    "ACEStep15SemanticHintsBlend": ACEStep15SemanticHintsBlend,
    # ACE-Step 1.5 guiders (model-level mask input)
    "ACEStep15NativeEditGuider": ACEStep15NativeEditGuiderNode,
    "ACEStep15NativeCoverGuider": ACEStep15NativeCoverGuiderNode,
    "ACEStep15NativeExtractGuider": ACEStep15NativeExtractGuiderNode,
    "ACEStep15NativeLegoGuider": ACEStep15NativeLegoGuiderNode,
    # ACE-Step 1.5 text encoder
    "ACEStep15TaskTextEncode": ACEStep15TaskTextEncodeNode,
    # ACE-Step 1.5 generation steering, musical controls, latent channel EQ, and keystone config
    "ACEStep15GenerationSteering": ACEStep15GenerationSteering,
    "ACEStep15MusicalControls": ACEStep15MusicalControls,
    "ACEStep15LatentChannelEQ": ACEStep15LatentChannelEQ,
    "ACEStep15KeystoneConfig": ACEStep15KeystoneConfig,
    # ACE-Step 1.5 model sampling override
    "ModelSamplingACEStep": ModelSamplingACEStep,
    **AUDIO_MASK_NODE_CLASS_MAPPINGS,  # Add audio mask nodes
}

# Display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    # ACE-Step 1.0 guiders
    "ACEStepRepaintGuider": "ACEStep 1.0 Repaint Guider BETA",
    "ACEStepExtendGuider": "ACEStep 1.0 Extend Guider BETA",
    "ACEStepHybridGuider": "ACEStep 1.0 Hybrid Guider BETA",
    # Utility nodes
    "ACEStepAnalyzeLatent": "ACEStep Analyze Latent",
    "ACEStepTimeRange": "ACEStep Time Range",
    "ACEStepMaskVisualizer": "ACEStep Mask Visualizer",
    "ACEStepAudioPostProcessor": "ACEStep Audio Post Processor",
    # ACE-Step 1.5 semantic extraction and blending
    "ACEStep15SemanticExtractor": "ACE-Step 1.5 Semantic Extractor",
    "ACEStep15SemanticHintsBlend": "ACE-Step 1.5 Semantic Hints Blend",
    # ACE-Step 1.5 guiders
    "ACEStep15NativeEditGuider": "ACE-Step 1.5 Edit Guider (Extend/Repaint)",
    "ACEStep15NativeCoverGuider": "ACE-Step 1.5 Cover Guider",
    "ACEStep15NativeExtractGuider": "ACE-Step 1.5 Extract Guider",
    "ACEStep15NativeLegoGuider": "ACE-Step 1.5 Lego Guider",
    # ACE-Step 1.5 text encoder
    "ACEStep15TaskTextEncode": "ACE-Step 1.5 Task Text Encode",
    # ACE-Step 1.5 generation steering, musical controls, latent channel EQ, and keystone config
    "ACEStep15GenerationSteering": "ACE-Step 1.5 Denoising Trajectory EQ (Experimental)",
    "ACEStep15MusicalControls": "ACE-Step 1.5 Denoising Trajectory Aggregator (Experimental)",
    "ACEStep15LatentChannelEQ": "ACE-Step 1.5 Latent Channel EQ (Experimental)",
    "ACEStep15KeystoneConfig": "ACE-Step 1.5 Keystone Config (Experimental)",
    # ACE-Step 1.5 model sampling override
    "ModelSamplingACEStep": "ACE-Step Model Sampling",
    **AUDIO_MASK_NODE_DISPLAY_NAME_MAPPINGS,
} 