import torch
import torch.nn.functional as F
import math
import os
import comfy.model_management
import folder_paths
from . import logger

# Module-level cache for silence_latent (loaded once per session)
_silence_latent_cache = None

# HuggingFace URL for silence_latent.pt
SILENCE_LATENT_URL = "https://huggingface.co/ACE-Step/Ace-Step1.5/resolve/main/acestep-v15-turbo/silence_latent.pt"


def get_silence_latent_path():
    """Get the path where silence_latent.pt should be stored."""
    models_dir = folder_paths.models_dir
    ace_step_dir = os.path.join(models_dir, "ace_step")
    return os.path.join(ace_step_dir, "silence_latent.pt")


def download_silence_latent():
    """Download silence_latent.pt from HuggingFace."""
    import urllib.request

    silence_latent_path = get_silence_latent_path()
    ace_step_dir = os.path.dirname(silence_latent_path)

    # Create directory if it doesn't exist
    os.makedirs(ace_step_dir, exist_ok=True)

    logger.info(f"[ACE-Step] Downloading silence_latent.pt from HuggingFace...")
    logger.info(f"[ACE-Step]   URL: {SILENCE_LATENT_URL}")
    logger.info(f"[ACE-Step]   Destination: {silence_latent_path}")

    try:
        urllib.request.urlretrieve(SILENCE_LATENT_URL, silence_latent_path)
        logger.info(f"[ACE-Step] Download complete!")
        return True
    except Exception as e:
        logger.info(f"[ACE-Step] ERROR: Failed to download silence_latent.pt: {e}")
        return False


def load_silence_latent(verbose=True):
    """
    Load the silence_latent tensor, downloading if necessary.

    The silence_latent is a LEARNED tensor that represents "silence" in the latent space.
    It is NOT zeros - it's a specific tensor that the model was trained with.
    This tells the model to "generate new content here".

    Returns:
        torch.Tensor: silence_latent in shape [1, T, D] (transposed from file)
    """
    global _silence_latent_cache

    # Return cached version if available
    if _silence_latent_cache is not None:
        if verbose:
            logger.info(f"[ACE-Step] Using cached silence_latent, shape: {_silence_latent_cache.shape}")
        return _silence_latent_cache

    silence_latent_path = get_silence_latent_path()

    # Download if not exists
    if not os.path.exists(silence_latent_path):
        success = download_silence_latent()
        if not success:
            raise RuntimeError(
                f"Failed to download silence_latent.pt. Please manually download from:\n"
                f"  {SILENCE_LATENT_URL}\n"
                f"and place it at:\n"
                f"  {silence_latent_path}"
            )

    # Load the tensor
    if verbose:
        logger.info(f"[ACE-Step] Loading silence_latent from {silence_latent_path}")
    silence_latent = torch.load(silence_latent_path, map_location="cpu", weights_only=True)

    # Transpose from [1, D, T] to [1, T, D] as per reference implementation
    # handler.py line 465: self.silence_latent = torch.load(silence_latent_path).transpose(1, 2)
    silence_latent = silence_latent.transpose(1, 2)

    if verbose:
        logger.info(f"[ACE-Step] silence_latent loaded, shape: {silence_latent.shape}")

    # Cache for future use
    _silence_latent_cache = silence_latent

    return silence_latent


def extract_semantic_hints(model, source_latent, verbose=True):
    """
    Extract semantic hints from source audio latents for Cover/Extract tasks.

    This extracts the semantic structure of the audio (rhythm, melody, harmony)
    while abstracting away the timbre. Used by Cover and Extract guiders.

    Args:
        model: ComfyUI ModelPatcher wrapping the ACE-Step 1.5 model
        source_latent: Source audio latent in ComfyUI format [B, 64, T]
        verbose: Whether to print diagnostic information

    Returns:
        torch.Tensor: Semantic hints in ComfyUI format [B, 64, T]
    """
    print(f"[DIAG_SEMANTIC_EXTRACT] input latent shape={source_latent.shape}, mean={source_latent.mean():.4f}, std={source_latent.std():.4f}")

    prefix = "[SEMANTIC_EXTRACT]"

    # Validate v1.5 shape: (batch, 64, length)
    if len(source_latent.shape) != 3 or source_latent.shape[1] != 64:
        raise ValueError(f"ACE-Step 1.5 requires latent shape (batch, 64, length), got {source_latent.shape}")

    # Get the diffusion model (AceStepConditionGenerationModel)
    diffusion_model = model.model.diffusion_model

    if not hasattr(diffusion_model, 'tokenizer') or not hasattr(diffusion_model, 'detokenizer'):
        raise RuntimeError(
            "Model does not have tokenizer/detokenizer. "
            "Make sure you're using an ACE-Step 1.5 model."
        )

    # Load model to GPU
    comfy.model_management.load_model_gpu(model)

    device = comfy.model_management.get_torch_device()
    dtype = model.model.get_dtype()

    # Move source tensor to model device/dtype
    # Source is in ComfyUI format: [B, D, T] = [B, 64, length]
    # Tokenizer expects: [B, T, D] = [B, length, 64]
    source_on_device = source_latent.to(device=device, dtype=dtype)
    source_transposed = source_on_device.movedim(-1, -2)  # [B, T, D]

    if verbose:
        logger.debug(f"{prefix} Extracting from source shape: {source_latent.shape}")
        logger.debug(f"{prefix}   source stats: mean={source_transposed.mean():.4f}, std={source_transposed.std():.4f}")

    # Verify weights are on GPU
    tokenizer_device = next(diffusion_model.tokenizer.parameters()).device
    if verbose:
        logger.debug(f"{prefix}   tokenizer on: {tokenizer_device}, target: {device}")

    # Step 1: Tokenize - get quantized embeddings at 5Hz
    with torch.no_grad():
        quantized, indices = diffusion_model.tokenizer.tokenize(source_transposed)

    if verbose:
        logger.debug(f"{prefix}   quantized shape: {quantized.shape}")
        logger.debug(f"{prefix}   quantized stats: mean={quantized.mean():.4f}, std={quantized.std():.4f}")

    # Step 2: Detokenize - upsample from 5Hz to 25Hz
    # Use quantized directly (not get_output_from_indices) to avoid redundant computation
    with torch.no_grad():
        lm_hints = diffusion_model.detokenizer(quantized)

    if verbose:
        logger.debug(f"{prefix}   lm_hints shape: {lm_hints.shape}")
        logger.debug(f"{prefix}   lm_hints stats: mean={lm_hints.mean():.4f}, std={lm_hints.std():.4f}")
        if lm_hints.std() < 0.01:
            logger.warning(f"{prefix}   Very low variance in semantic hints!")

    # Transpose back to ComfyUI format: [B, T, D] → [B, D, T]
    semantic_hints = lm_hints.movedim(-1, -2)

    print(f"[DIAG_SEMANTIC_EXTRACT] output hints shape={semantic_hints.shape}, mean={semantic_hints.mean():.4f}, std={semantic_hints.std():.4f}")

    return semantic_hints


class ACEStepLatentUtils:
    """Utility functions for ACEStep audio latent manipulation"""

    # Version constants
    V1_0 = "v1.0"
    V1_5 = "v1.5"

    # Frame rates: frames per second
    V1_0_FPS = 44100 / 512 / 8  # ≈ 10.77 fps
    V1_5_FPS = 48000 / 1920     # = 25 fps

    @staticmethod
    def detect_version(latent):
        """Detect ACE-Step version from latent shape
        v1.0: (batch, 8, 16, length) - 4D
        v1.5: (batch, 64, length) - 3D
        """
        if len(latent.shape) == 4 and latent.shape[1] == 8 and latent.shape[2] == 16:
            return ACEStepLatentUtils.V1_0
        elif len(latent.shape) == 3 and latent.shape[1] == 64:
            return ACEStepLatentUtils.V1_5
        else:
            return None

    @staticmethod
    def get_fps(version):
        """Get frames per second for a given version"""
        if version == ACEStepLatentUtils.V1_5:
            return ACEStepLatentUtils.V1_5_FPS
        return ACEStepLatentUtils.V1_0_FPS

    @staticmethod
    def time_to_frame_index(time_seconds, version=None):
        """Convert time in seconds to ACE latent frame index
        v1.0: 1 second = 44100 / 512 / 8 ≈ 10.77 frames
        v1.5: 1 second = 48000 / 1920 = 25 frames
        """
        fps = ACEStepLatentUtils.get_fps(version)
        return int(time_seconds * fps)

    @staticmethod
    def frame_index_to_time(frame_index, version=None):
        """Convert ACE latent frame index to time in seconds"""
        fps = ACEStepLatentUtils.get_fps(version)
        return frame_index / fps

    @staticmethod
    def get_latent_length(latent):
        """Get the time dimension length from latent (last dim for both versions)"""
        return latent.shape[-1]

    @staticmethod
    def validate_ace_latent_shape(latent):
        """Validate that latent has correct ACE audio shape (v1.0 or v1.5)"""
        version = ACEStepLatentUtils.detect_version(latent)
        if version is None:
            if len(latent.shape) == 4:
                raise ValueError(f"ACE v1.0 latent must have shape (batch, 8, 16, length), got {latent.shape}")
            elif len(latent.shape) == 3:
                raise ValueError(f"ACE v1.5 latent must have shape (batch, 64, length), got {latent.shape}")
            else:
                raise ValueError(f"ACE latent must be 3D (v1.5) or 4D (v1.0), got {len(latent.shape)}D")
        return version
    
    @staticmethod
    def create_repaint_mask(latent_shape, start_frame, end_frame, version=None):
        """Create a mask for repainting a specific region

        Args:
            latent_shape: (batch, 8, 16, length) for v1.0 or (batch, 64, length) for v1.5
            start_frame: Frame index to start repainting
            end_frame: Frame index to end repainting
            version: ACE-Step version (auto-detected if None)

        Returns:
            torch.Tensor: Binary mask with 1.0 in repaint region
        """
        length = latent_shape[-1]

        # Clamp frame indices to valid range
        start_frame = max(0, start_frame)
        end_frame = min(length, end_frame)

        if start_frame >= end_frame:
            raise ValueError(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")

        mask = torch.zeros(latent_shape)
        if len(latent_shape) == 4:  # v1.0
            mask[:, :, :, start_frame:end_frame] = 1.0
        else:  # v1.5
            mask[:, :, start_frame:end_frame] = 1.0

        return mask

    @staticmethod
    def create_extend_mask(source_shape, left_frames, right_frames):
        """Create a mask for extension regions

        Args:
            source_shape: (batch, 8, 16, length) for v1.0 or (batch, 64, length) for v1.5
            left_frames: Number of frames to extend on the left
            right_frames: Number of frames to extend on the right

        Returns:
            torch.Tensor: Binary mask with 1.0 in extension regions
        """
        source_length = source_shape[-1]
        total_length = source_length + left_frames + right_frames

        if len(source_shape) == 4:  # v1.0
            batch_size, channels, height, _ = source_shape
            mask = torch.zeros((batch_size, channels, height, total_length))
            if left_frames > 0:
                mask[:, :, :, :left_frames] = 1.0
            if right_frames > 0:
                mask[:, :, :, -right_frames:] = 1.0
        else:  # v1.5
            batch_size, channels, _ = source_shape
            mask = torch.zeros((batch_size, channels, total_length))
            if left_frames > 0:
                mask[:, :, :left_frames] = 1.0
            if right_frames > 0:
                mask[:, :, -right_frames:] = 1.0

        return mask

    @staticmethod
    def create_extend_region_mask(extended_shape, left_frames, right_frames, source_length):
        """Create a mask indicating which regions to generate vs preserve in extended latent

        Args:
            extended_shape: Shape of the extended latent
            left_frames: Number of frames added on the left
            right_frames: Number of frames added on the right
            source_length: Length of the original source latent

        Returns:
            torch.Tensor: Binary mask with 1.0 = generate (extension regions), 0.0 = preserve (source region)
        """
        mask = torch.zeros(extended_shape)

        if len(extended_shape) == 4:  # v1.0
            if left_frames > 0:
                mask[:, :, :, :left_frames] = 1.0
            if right_frames > 0:
                mask[:, :, :, -right_frames:] = 1.0
        else:  # v1.5
            if left_frames > 0:
                mask[:, :, :left_frames] = 1.0
            if right_frames > 0:
                mask[:, :, -right_frames:] = 1.0

        return mask
    
    @staticmethod
    def pad_latent_for_extend(latent, left_frames, right_frames, pad_value=0.0):
        """Pad latent for extension
        
        Args:
            latent: Source latent tensor (batch, 8, 16, length)
            left_frames: Frames to pad on left
            right_frames: Frames to pad on right
            pad_value: Value to use for padding
            
        Returns:
            torch.Tensor: Padded latent
        """
        if left_frames == 0 and right_frames == 0:
            return latent
            
        return F.pad(latent, (left_frames, right_frames), 'constant', pad_value)
    
    @staticmethod
    def blend_latents_with_mask(original, new, mask, blend_factor=1.0):
        """Blend two latents using a mask
        
        Args:
            original: Original latent tensor
            new: New latent tensor
            mask: Binary mask (1.0 = use new, 0.0 = use original)
            blend_factor: Overall blending strength
            
        Returns:
            torch.Tensor: Blended latent
        """
        mask = mask * blend_factor
        return original * (1 - mask) + new * mask
    
    @staticmethod
    def calculate_timestep_blend_factor(timestep, max_timestep=1000, 
                                      min_factor=0.1, max_factor=1.0):
        """Calculate blend factor based on timestep
        
        Higher timesteps (more noise) use more blending
        Lower timesteps (less noise) preserve more original content
        """
        normalized_timestep = timestep / max_timestep
        return min_factor + (max_factor - min_factor) * normalized_timestep
    
    @staticmethod
    def create_feather_mask(mask, feather_frames=2):
        """Create a feathered (soft) version of a binary mask

        Args:
            mask: Binary mask tensor (3D for v1.5 or 4D for v1.0)
            feather_frames: Number of frames to feather on each side

        Returns:
            torch.Tensor: Feathered mask with smooth transitions
        """
        if feather_frames <= 0:
            return mask

        # Create gaussian kernel for smoothing
        kernel_size = feather_frames * 2 + 1
        sigma = feather_frames / 3.0

        x = torch.arange(kernel_size, dtype=torch.float32) - feather_frames
        gaussian_kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        result = torch.zeros_like(mask)
        is_v1_5 = len(mask.shape) == 3

        if is_v1_5:
            # v1.5: (batch, channels, length)
            batch_size, channels, length = mask.shape
            kernel = gaussian_kernel.view(1, 1, -1)

            for b in range(batch_size):
                for c in range(channels):
                    signal = mask[b:b+1, c:c+1, :]
                    padded_signal = F.pad(signal, (feather_frames, feather_frames), 'constant', 0.0)
                    smoothed = F.conv1d(padded_signal, kernel, padding=0)
                    result[b, c, :] = smoothed[0, 0, :]
        else:
            # v1.0: (batch, channels, height, length)
            batch_size, channels, height, length = mask.shape
            kernel = gaussian_kernel.view(1, 1, 1, -1)

            for b in range(batch_size):
                for c in range(channels):
                    for h in range(height):
                        signal = mask[b:b+1, c:c+1, h:h+1, :]
                        padded_signal = F.pad(signal, (feather_frames, feather_frames), 'constant', 0.0)
                        smoothed = F.conv2d(padded_signal, kernel, padding=0)
                        result[b, c, h, :] = smoothed[0, 0, 0, :]

        return torch.clamp(result, 0.0, 1.0)
    
    @staticmethod
    def apply_repaint_with_timestep_blending(noise_pred, source_latent, repaint_mask, timestep, repaint_strength):
        """
        Apply repaint logic with timestep-aware blending
        
        Args:
            noise_pred: Predicted noise from the model
            source_latent: Original latent content
            repaint_mask: Binary mask indicating repaint region
            timestep: Current diffusion timestep
            repaint_strength: Strength of repainting (0.0 to 1.0)
        
        Returns:
            Modified noise prediction with repaint logic applied
        """
        device = noise_pred.device
        repaint_mask = repaint_mask.to(device)
        source_latent = source_latent.to(device)
        
        # Calculate timestep-aware blending factor
        # At early timesteps (high noise), apply less of the original content
        # At later timesteps (low noise), preserve more of the original content
        timestep_normalized = timestep / 1000.0  # Normalize to 0-1 range
        
        # Blend factor decreases as we get closer to final result (lower timestep)
        # This allows the repaint region to be more influenced by conditioning at early steps
        blend_factor = repaint_strength * (1.0 - timestep_normalized)
        
        # In the repaint region, blend between new generation and original content
        # Outside repaint region, preserve original content completely
        inverse_mask = 1.0 - repaint_mask
        
        # Apply repaint blending
        noise_pred_blended = (
            noise_pred * repaint_mask * (1.0 - blend_factor) +  # New generation in repaint region
            source_latent * repaint_mask * blend_factor +       # Original content in repaint region  
            noise_pred * inverse_mask                           # Preserve outside repaint region
        )
        
        return noise_pred_blended
    
    @staticmethod
    def apply_extend_preservation(noise_pred, extended_latent, region_mask, timestep):
        """
        Apply extend logic to preserve original content and generate in extension regions
        
        Args:
            noise_pred: Predicted noise from the model
            extended_latent: Extended latent with padding
            region_mask: Mask indicating which regions to preserve vs generate
            timestep: Current diffusion timestep
        
        Returns:
            Modified noise prediction with extend logic applied
        """
        device = noise_pred.device
        region_mask = region_mask.to(device)
        extended_latent = extended_latent.to(device)
        
        # region_mask: 1.0 = generate (extension regions), 0.0 = preserve (original content)
        preservation_mask = 1.0 - region_mask  # Flip to get preservation mask
        
        # At early timesteps, allow more mixing to help continuity
        # At later timesteps, strictly preserve original content
        timestep_normalized = timestep / 1000.0
        preservation_strength = 0.3 + 0.7 * (1.0 - timestep_normalized)
        
        # Preserve original content in source region, generate in extension regions
        noise_pred_extended = (
            noise_pred * region_mask +                                    # Generate in extension regions
            extended_latent * preservation_mask * preservation_strength + # Preserve original content
            noise_pred * preservation_mask * (1.0 - preservation_strength) # Allow some blending
        )
        
        return noise_pred_extended

class ACEStepRepaintHelper:
    """Helper class for repaint operations"""

    def __init__(self, source_latents, start_time, end_time,
                 repaint_strength=0.7, feather_time=0.1):
        """
        Args:
            source_latents: Original audio latents (v1.0 or v1.5 shape)
            start_time: Start time in seconds for repaint region
            end_time: End time in seconds for repaint region
            repaint_strength: Strength of repainting (0.0 = no repaint, 1.0 = full repaint)
            feather_time: Time in seconds for feathering edges
        """
        self.version = ACEStepLatentUtils.validate_ace_latent_shape(source_latents)

        self.source_latents = source_latents
        self.start_frame = ACEStepLatentUtils.time_to_frame_index(start_time, self.version)
        self.end_frame = ACEStepLatentUtils.time_to_frame_index(end_time, self.version)
        self.repaint_strength = repaint_strength
        self.feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_time, self.version)

        # Create masks
        self.repaint_mask = ACEStepLatentUtils.create_repaint_mask(
            source_latents.shape, self.start_frame, self.end_frame, self.version)

        # Apply feathering for smooth transitions
        if self.feather_frames > 0:
            self.repaint_mask = ACEStepLatentUtils.create_feather_mask(
                self.repaint_mask, self.feather_frames)
    
    def apply_repaint_blend(self, new_latents, timestep, max_timestep=1000):
        """Apply repaint blending based on timestep"""
        # Calculate dynamic blend factor based on timestep
        timestep_factor = ACEStepLatentUtils.calculate_timestep_blend_factor(
            timestep, max_timestep)
        
        # Combine with user-defined repaint strength
        blend_factor = self.repaint_strength * timestep_factor
        
        # Blend latents
        return ACEStepLatentUtils.blend_latents_with_mask(
            self.source_latents, new_latents, self.repaint_mask, blend_factor)

class ACEStepExtendHelper:
    """Helper class for extend operations"""

    def __init__(self, source_latents, extend_left_time=0, extend_right_time=0):
        """
        Args:
            source_latents: Original audio latents (v1.0 or v1.5 shape)
            extend_left_time: Time in seconds to extend before the audio
            extend_right_time: Time in seconds to extend after the audio
        """
        self.version = ACEStepLatentUtils.validate_ace_latent_shape(source_latents)

        self.source_latents = source_latents
        self.left_frames = ACEStepLatentUtils.time_to_frame_index(extend_left_time, self.version)
        self.right_frames = ACEStepLatentUtils.time_to_frame_index(extend_right_time, self.version)

        # Create extended latent with zero padding
        self.extended_latents = ACEStepLatentUtils.pad_latent_for_extend(
            source_latents, self.left_frames, self.right_frames)

        # Create extension mask
        self.extension_mask = ACEStepLatentUtils.create_extend_mask(
            source_latents.shape, self.left_frames, self.right_frames)
    
    def get_extended_shape(self):
        """Get the shape of the extended latent"""
        return self.extended_latents.shape
    
    def apply_extension_blend(self, new_latents):
        """Apply extension blending - preserve original, generate in extension regions"""
        return ACEStepLatentUtils.blend_latents_with_mask(
            self.extended_latents, new_latents, self.extension_mask, 1.0) 