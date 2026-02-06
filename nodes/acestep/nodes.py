import torch
import os
import comfy.model_management
import comfy.samplers
import folder_paths

# Import patches module to apply ACE-Step 1.5 patches
from . import patches

from .ace_step_guiders import (
    ACEStepRepaintGuider, ACEStepExtendGuider, ACEStepHybridGuider,
    ACEStep15NativeEditGuider,
    ACEStep15NativeCoverGuider, ACEStep15NativeExtractGuider, ACEStep15NativeLegoGuider
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


class ACEStep15SilenceLatentLoader:
    """
    Loads the ACE-Step 1.5 silence_latent tensor required for repaint and lego tasks.

    The silence_latent is a LEARNED tensor that represents "silence" in the latent space.
    It is NOT zeros - it's a specific tensor that the model was trained with.

    Downloads automatically from HuggingFace on first use.
    """

    # HuggingFace URL for silence_latent.pt
    SILENCE_LATENT_URL = "https://huggingface.co/ACE-Step/Ace-Step1.5/resolve/main/acestep-v15-turbo/silence_latent.pt"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }

    RETURN_TYPES = ("SILENCE_LATENT",)
    RETURN_NAMES = ("silence_latent",)
    FUNCTION = "load"
    CATEGORY = "audio/acestep"

    @classmethod
    def get_silence_latent_path(cls):
        """Get the path where silence_latent.pt should be stored."""
        # Use ComfyUI's models directory structure
        models_dir = folder_paths.models_dir
        ace_step_dir = os.path.join(models_dir, "ace_step")
        return os.path.join(ace_step_dir, "silence_latent.pt")

    @classmethod
    def download_silence_latent(cls):
        """Download silence_latent.pt from HuggingFace."""
        import urllib.request

        silence_latent_path = cls.get_silence_latent_path()
        ace_step_dir = os.path.dirname(silence_latent_path)

        # Create directory if it doesn't exist
        os.makedirs(ace_step_dir, exist_ok=True)

        print(f"[ACE-Step] Downloading silence_latent.pt from HuggingFace...")
        print(f"[ACE-Step]   URL: {cls.SILENCE_LATENT_URL}")
        print(f"[ACE-Step]   Destination: {silence_latent_path}")

        try:
            urllib.request.urlretrieve(cls.SILENCE_LATENT_URL, silence_latent_path)
            print(f"[ACE-Step] Download complete!")
            return True
        except Exception as e:
            print(f"[ACE-Step] ERROR: Failed to download silence_latent.pt: {e}")
            return False

    def load(self):
        """Load the silence_latent tensor, downloading if necessary."""
        silence_latent_path = self.get_silence_latent_path()

        # Download if not exists
        if not os.path.exists(silence_latent_path):
            success = self.download_silence_latent()
            if not success:
                raise RuntimeError(
                    f"Failed to download silence_latent.pt. Please manually download from:\n"
                    f"  {self.SILENCE_LATENT_URL}\n"
                    f"and place it at:\n"
                    f"  {silence_latent_path}"
                )

        # Load the tensor
        print(f"[ACE-Step] Loading silence_latent from {silence_latent_path}")
        silence_latent = torch.load(silence_latent_path, map_location="cpu", weights_only=True)

        # Transpose from [1, D, T] to [1, T, D] as per reference implementation
        # handler.py line 465: self.silence_latent = torch.load(silence_latent_path).transpose(1, 2)
        silence_latent = silence_latent.transpose(1, 2)

        print(f"[ACE-Step] silence_latent loaded, shape: {silence_latent.shape}")

        return (silence_latent,)


class ACEStep15SemanticExtractor:
    """
    Extracts semantic tokens from source audio latents for use with Cover/Extract tasks.

    The ACE-Step 1.5 model uses semantic tokens (lm_hints) as structural guidance
    for tasks like Cover and Extract. This node extracts those tokens from VAE latents.

    Flow: VAE latents → tokenizer → indices → quantizer → detokenizer → semantic_hints

    Use the output with ACE-Step 1.5 Cover Guider or Extract Guider for proper
    style transfer behavior.
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
        source_tensor = source_latents["samples"]

        # Validate v1.5 shape: (batch, 64, length)
        if len(source_tensor.shape) != 3 or source_tensor.shape[1] != 64:
            raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_tensor.shape}")

        print(f"[ACE15_SEMANTIC_EXTRACTOR] Extracting semantic hints")
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   source_tensor.shape: {source_tensor.shape}")

        # Debug: Print model structure
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   model type: {type(model)}")
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   model.__class__.__name__: {model.__class__.__name__}")
        if hasattr(model, 'model'):
            print(f"[ACE15_SEMANTIC_EXTRACTOR]   model.model type: {type(model.model)}")
            print(f"[ACE15_SEMANTIC_EXTRACTOR]   model.model.__class__.__name__: {model.model.__class__.__name__}")
            if hasattr(model.model, 'diffusion_model'):
                print(f"[ACE15_SEMANTIC_EXTRACTOR]   model.model.diffusion_model type: {type(model.model.diffusion_model)}")
                print(f"[ACE15_SEMANTIC_EXTRACTOR]   model.model.diffusion_model.__class__.__name__: {model.model.diffusion_model.__class__.__name__}")

        # Get the diffusion model (AceStepConditionGenerationModel)
        # It has .tokenizer and .detokenizer attributes
        diffusion_model = model.model.diffusion_model

        if not hasattr(diffusion_model, 'tokenizer') or not hasattr(diffusion_model, 'detokenizer'):
            raise RuntimeError(
                "Model does not have tokenizer/detokenizer. "
                "Make sure you're using an ACE-Step 1.5 model."
            )

        # Load model to GPU using ComfyUI's model management
        # Pass the ModelPatcher (model), not the raw model (model.model)
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   Loading model to GPU...")
        comfy.model_management.load_model_gpu(model)

        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()

        print(f"[ACE15_SEMANTIC_EXTRACTOR]   device: {device}, dtype: {dtype}")

        # Move source tensor to model device/dtype
        # Source is in ComfyUI format: [B, D, T] = [B, 64, length]
        # Tokenizer expects: [B, T, D] = [B, length, 64]
        source_on_device = source_tensor.to(device=device, dtype=dtype)
        source_transposed = source_on_device.movedim(-1, -2)  # [B, T, D]

        print(f"[ACE15_SEMANTIC_EXTRACTOR]   source_transposed.shape: {source_transposed.shape}")
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   source stats: mean={source_transposed.mean():.4f}, std={source_transposed.std():.4f}, min={source_transposed.min():.4f}, max={source_transposed.max():.4f}")

        # Verify tokenizer/detokenizer weights are on GPU
        tokenizer_device = next(diffusion_model.tokenizer.parameters()).device
        detokenizer_device = next(diffusion_model.detokenizer.parameters()).device
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   tokenizer weights on: {tokenizer_device}")
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   detokenizer weights on: {detokenizer_device}")

        # Step 1: Tokenize - get quantized embeddings at 5Hz
        # tokenizer.tokenize() handles the window pooling internally
        # Returns: quantized [B, T/5, hidden_size], indices [B, T/5, num_quantizers]
        with torch.no_grad():
            quantized, indices = diffusion_model.tokenizer.tokenize(source_transposed)

        print(f"[ACE15_SEMANTIC_EXTRACTOR]   quantized.shape: {quantized.shape}")
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   quantized stats: mean={quantized.mean():.4f}, std={quantized.std():.4f}, min={quantized.min():.4f}, max={quantized.max():.4f}")
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   indices.shape: {indices.shape}")

        # Step 2: Detokenize - upsample from 5Hz to 25Hz
        # FIX: Use quantized directly instead of going through get_output_from_indices
        # This avoids potential precision loss from the redundant index lookup
        with torch.no_grad():
            lm_hints = diffusion_model.detokenizer(quantized)

        print(f"[ACE15_SEMANTIC_EXTRACTOR]   lm_hints.shape: {lm_hints.shape}")
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   lm_hints stats: mean={lm_hints.mean():.4f}, std={lm_hints.std():.4f}, min={lm_hints.min():.4f}, max={lm_hints.max():.4f}")

        # Transpose back to ComfyUI format: [B, T, D] → [B, D, T]
        semantic_hints = lm_hints.movedim(-1, -2)
        print(f"[ACE15_SEMANTIC_EXTRACTOR]   semantic_hints.shape (output): {semantic_hints.shape}")

        # Move to CPU for storage
        semantic_hints = semantic_hints.cpu()

        return (semantic_hints,)


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

        print(f"[EXTEND_GUIDER_NODE] get_guider called")
        print(f"[EXTEND_GUIDER_NODE]   extend_left_time: {extend_left_time}")
        print(f"[EXTEND_GUIDER_NODE]   extend_right_time: {extend_right_time}")

        if extend_left_time == 0 and extend_right_time == 0:
            raise ValueError("At least one of extend_left_time or extend_right_time must be > 0")

        if not validate_audio_latent(source_latents):
            raise ValueError("source_latents must be audio latents (from VAEEncodeAudio or EmptyAceStepLatentAudio)")

        # Extract latent tensor
        latent_tensor = source_latents["samples"]
        print(f"[EXTEND_GUIDER_NODE]   latent_tensor.shape: {latent_tensor.shape}")
        version = ACEStepLatentUtils.detect_version(latent_tensor)
        print(f"[EXTEND_GUIDER_NODE]   detected version: {version}")

        # Create and return the extend guider
        guider = ACEStepExtendGuider(
            model, positive, negative, cfg,
            latent_tensor, extend_left_time, extend_right_time
        )
        print(f"[EXTEND_GUIDER_NODE]   guider.is_v1_5: {guider.is_v1_5}")
        print(f"[EXTEND_GUIDER_NODE]   guider.extended_latent.shape: {guider.extended_latent.shape}")

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

    IMPORTANT: Requires silence_latent from ACEStep15SilenceLatentLoader.
    The silence_latent is a learned tensor that tells the model to generate
    new content in the specified regions.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "source_latents": ("LATENT",),
                "silence_latent": ("SILENCE_LATENT",),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            },
            "optional": {
                "extend_left_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.1}),
                "extend_right_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.1}),
                "repaint_start_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
                "repaint_end_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("GUIDER", "LATENT")
    RETURN_NAMES = ("guider", "output_latent")
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, source_latents, silence_latent, cfg,
                   extend_left_seconds=0.0, extend_right_seconds=0.0,
                   repaint_start_seconds=-1.0, repaint_end_seconds=-1.0):

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

        print(f"[ACE15_EDIT_NODE] Creating guider")
        print(f"[ACE15_EDIT_NODE]   source_tensor.shape: {source_tensor.shape}")
        print(f"[ACE15_EDIT_NODE]   silence_latent.shape: {silence_latent.shape}")
        if has_extend:
            print(f"[ACE15_EDIT_NODE]   extend: left={extend_left_seconds}s, right={extend_right_seconds}s")
        if has_repaint:
            print(f"[ACE15_EDIT_NODE]   repaint: {repaint_start_seconds}s - {repaint_end_seconds}s")

        # Create the unified guider
        guider = ACEStep15NativeEditGuider(
            model, positive, negative, cfg,
            source_tensor,
            silence_latent,
            extend_left_seconds=extend_left_seconds,
            extend_right_seconds=extend_right_seconds,
            repaint_start_seconds=repaint_start,
            repaint_end_seconds=repaint_end
        )

        # Return the working latent (may be extended or same as source)
        output_latent = {"samples": guider.working_latent, "type": "audio"}

        return (guider, output_latent)


class ACEStep15NativeCoverGuiderNode:
    """
    Creates an ACE-Step 1.5 cover guider for style transfer/regeneration.

    Uses semantic tokens from the source audio as structural guidance while
    generating new content with a different style/timbre.

    IMPORTANT: For proper cover behavior, connect semantic_hints from ACEStep15SemanticExtractor.
    Without semantic_hints, falls back to using raw VAE latents (less accurate style transfer).
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
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg, source_latents, semantic_hints=None):
        source_tensor = source_latents["samples"]

        # Validate v1.5 shape: (batch, 64, length)
        if len(source_tensor.shape) != 3 or source_tensor.shape[1] != 64:
            raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_tensor.shape}")

        print(f"[ACE15_COVER_NODE] Creating guider")
        print(f"[ACE15_COVER_NODE]   source_tensor.shape: {source_tensor.shape}")
        if semantic_hints is not None:
            print(f"[ACE15_COVER_NODE]   semantic_hints.shape: {semantic_hints.shape}")
        else:
            print(f"[ACE15_COVER_NODE]   WARNING: No semantic_hints provided - using fallback mode")

        guider = ACEStep15NativeCoverGuider(
            model, positive, negative, cfg, source_tensor, semantic_hints
        )

        return (guider,)


class ACEStep15NativeExtractGuiderNode:
    """
    Creates an ACE-Step 1.5 extract guider for extracting specific tracks.

    Extracts a specific track (vocals, drums, bass, etc.) from the source audio.
    Use with the ACEStep15TaskTextEncode node with task_type="extract".

    IMPORTANT: For proper extract behavior, connect semantic_hints from ACEStep15SemanticExtractor.
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
                "track_name": (["vocals", "drums", "bass", "guitar", "keyboard", "strings",
                               "percussion", "synth", "fx", "brass", "woodwinds", "backing_vocals"],),
            },
            "optional": {
                "semantic_hints": ("SEMANTIC_HINTS",),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg, source_latents, track_name, semantic_hints=None):
        source_tensor = source_latents["samples"]

        # Validate v1.5 shape: (batch, 64, length)
        if len(source_tensor.shape) != 3 or source_tensor.shape[1] != 64:
            raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_tensor.shape}")

        print(f"[ACE15_EXTRACT_NODE] Creating guider")
        print(f"[ACE15_EXTRACT_NODE]   source_tensor.shape: {source_tensor.shape}")
        print(f"[ACE15_EXTRACT_NODE]   track_name: {track_name}")
        if semantic_hints is not None:
            print(f"[ACE15_EXTRACT_NODE]   semantic_hints.shape: {semantic_hints.shape}")

        guider = ACEStep15NativeExtractGuider(
            model, positive, negative, cfg, source_tensor, track_name, semantic_hints
        )

        return (guider,)


class ACEStep15NativeLegoGuiderNode:
    """
    Creates an ACE-Step 1.5 lego guider for generating specific tracks in a region.

    Generates a specific track (vocals, drums, etc.) within a time region while
    preserving the rest. Useful for adding instruments to existing audio.
    Use with the ACEStep15TaskTextEncode node with task_type="lego".

    IMPORTANT: Requires silence_latent from ACEStep15SilenceLatentLoader.
    The silence_latent is a learned tensor that tells the model to generate
    new content in the lego region.
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
                "silence_latent": ("SILENCE_LATENT",),
                "track_name": (["vocals", "drums", "bass", "guitar", "keyboard", "strings",
                               "percussion", "synth", "fx", "brass", "woodwinds", "backing_vocals"],),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "end_seconds": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg, source_latents,
                   silence_latent, track_name, start_seconds, end_seconds):
        if start_seconds >= end_seconds:
            raise ValueError(f"start_seconds ({start_seconds}) must be less than end_seconds ({end_seconds})")

        source_tensor = source_latents["samples"]

        # Validate v1.5 shape: (batch, 64, length)
        if len(source_tensor.shape) != 3 or source_tensor.shape[1] != 64:
            raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_tensor.shape}")

        print(f"[ACE15_LEGO_NODE] Creating guider")
        print(f"[ACE15_LEGO_NODE]   source_tensor.shape: {source_tensor.shape}")
        print(f"[ACE15_LEGO_NODE]   silence_latent.shape: {silence_latent.shape}")
        print(f"[ACE15_LEGO_NODE]   track_name: {track_name}")
        print(f"[ACE15_LEGO_NODE]   region: {start_seconds}s - {end_seconds}s")

        guider = ACEStep15NativeLegoGuider(
            model, positive, negative, cfg, source_tensor,
            silence_latent, track_name, start_seconds, end_seconds
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

    # Valid keyscales from reference: 7 notes × 3 accidentals (plain, #, b) × 2 modes = 42 combinations
    # Using ASCII accidentals for compatibility
    KEYSCALE_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    KEYSCALE_ACCIDENTALS = ['', '#', 'b']
    KEYSCALE_MODES = ['major', 'minor']

    # Generate all valid keyscales
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
                "task_type": (["text2music", "repaint", "cover", "extract", "lego"],),
            },
            "optional": {
                "track_name": (["", "vocals", "drums", "bass", "guitar", "keyboard", "strings",
                               "percussion", "synth", "fx", "brass", "woodwinds", "backing_vocals"],
                              {"default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "bpm": ("INT", {"default": 120, "min": 30, "max": 300}),
                "duration": ("INT", {"default": 60, "min": 10, "max": 600}),
                # Use COMBO type to accept connections from AudioInfo detected_key output
                "keyscale": ("COMBO", {"default": "C major", "options": s.VALID_KEYSCALES}),
                "timesignature": ("COMBO", {"default": "4", "options": s.VALID_TIME_SIGNATURES}),
                "language": ("COMBO", {"default": "en", "options": s.VALID_LANGUAGES}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, task_type, track_name="", lyrics="", bpm=120,
               duration=60, keyscale="C major", timesignature="4", language="en", seed=0):
        # Validate track_name for extract/lego tasks
        if task_type in ["extract", "lego"] and not track_name:
            print(f"[ACE15_TEXT_ENCODE] Warning: track_name not specified for {task_type} task, using default instruction")

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
        }

        print(f"[ACE15_TEXT_ENCODE] Encoding with task_type={task_type}, track_name={track_name or 'N/A'}")

        # Import to get task instruction for verification
        from .patches import get_task_instruction, is_patched
        instruction = get_task_instruction(task_type, track_name if track_name else None)
        print(f"[ACE15_TEXT_ENCODE]   patches applied: {is_patched()}")
        print(f"[ACE15_TEXT_ENCODE]   task instruction: {instruction}")

        # Use the patched tokenize_with_weights which accepts task_type
        # encode_from_tokens_scheduled returns conditioning directly (list format)
        tokens = clip.tokenize(text, **kwargs)

        # Verify the tokens contain task_type info (patched tokenizer should include this)
        if isinstance(tokens, dict):
            if "task_type" in tokens:
                print(f"[ACE15_TEXT_ENCODE]   tokens contain task_type: {tokens.get('task_type')}")
            else:
                print(f"[ACE15_TEXT_ENCODE]   WARNING: tokens dict does not contain task_type - patch may not be applied!")
                print(f"[ACE15_TEXT_ENCODE]   tokens keys: {list(tokens.keys())}")

        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # For negative conditioning, use ComfyUI's ConditioningZeroOut node
        # This matches the standard ACE-Step workflow pattern

        return (conditioning,)


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
    # ACE-Step 1.5 loaders
    "ACEStep15SilenceLatentLoader": ACEStep15SilenceLatentLoader,
    # ACE-Step 1.5 semantic extraction
    "ACEStep15SemanticExtractor": ACEStep15SemanticExtractor,
    # ACE-Step 1.5 guiders (model-level mask input)
    "ACEStep15NativeEditGuider": ACEStep15NativeEditGuiderNode,
    "ACEStep15NativeCoverGuider": ACEStep15NativeCoverGuiderNode,
    "ACEStep15NativeExtractGuider": ACEStep15NativeExtractGuiderNode,
    "ACEStep15NativeLegoGuider": ACEStep15NativeLegoGuiderNode,
    # ACE-Step 1.5 text encoder
    "ACEStep15TaskTextEncode": ACEStep15TaskTextEncodeNode,
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
    # ACE-Step 1.5 loaders
    "ACEStep15SilenceLatentLoader": "ACE-Step 1.5 Load Silence Latent",
    # ACE-Step 1.5 semantic extraction
    "ACEStep15SemanticExtractor": "ACE-Step 1.5 Semantic Extractor",
    # ACE-Step 1.5 guiders
    "ACEStep15NativeEditGuider": "ACE-Step 1.5 Edit Guider (Extend/Repaint)",
    "ACEStep15NativeCoverGuider": "ACE-Step 1.5 Cover Guider",
    "ACEStep15NativeExtractGuider": "ACE-Step 1.5 Extract Guider",
    "ACEStep15NativeLegoGuider": "ACE-Step 1.5 Lego Guider",
    # ACE-Step 1.5 text encoder
    "ACEStep15TaskTextEncode": "ACE-Step 1.5 Task Text Encode",
    **AUDIO_MASK_NODE_DISPLAY_NAME_MAPPINGS,
} 