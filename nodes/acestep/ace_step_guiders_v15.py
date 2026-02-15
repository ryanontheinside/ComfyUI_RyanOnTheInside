"""
ACE-Step 1.5 guiders for ComfyUI.

These guiders inject chunk_masks, src_latents, is_covers, and semantic hints
into the model's forward() method via model wrapping.

Task types: edit (extend+repaint), cover, extract, lego.
"""

import torch
import comfy.samplers
from . import logger
from .ace_step_guiders_common import _apply_apg_guidance, _inject_reference_audio


# =============================================================================
# ACE-Step 1.5 Base Guider
# =============================================================================

class ACEStep15BaseGuider(comfy.samplers.CFGGuider):
    """Base class for all ACE-Step 1.5 guiders.

    Provides shared model wrapper injection (chunk_masks, src_latents, reference audio),
    APG guidance setup, and sampling skeleton. Subclasses override hooks to customize
    task-specific conditioning.
    """

    def __init__(self, model, positive, negative, cfg, source_latent, reference_latent=None):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(cfg)
        self.source_latent = source_latent
        self.reference_latent = reference_latent
        self._wrapper_applied = False

    def _get_extra_cond(self, input_batch_size, device, dtype):
        """Override in subclass to add task-specific conditioning (is_covers, semantic_hints)."""
        return {}

    def _get_wrapper_log_tag(self):
        """Override to provide log tag for one-time debug logging."""
        return "ACE15_BASE"

    def _apply_model_wrapper(self):
        if self._wrapper_applied:
            return
        chunk_masks = self.chunk_masks
        src_latents = self.src_latents
        reference_latent = self.reference_latent
        log_tag = self._get_wrapper_log_tag()
        extra_cond_fn = self._get_extra_cond
        self.model_patcher.set_model_unet_function_wrapper(None)
        logged = False

        def model_function_wrapper(apply_model, args):
            nonlocal logged
            c = args["c"].copy()
            device = args["input"].device
            dtype = args["input"].dtype
            batch = args["input"].shape[0]
            cm = chunk_masks.to(device=device, dtype=dtype)
            sl = src_latents.to(device=device, dtype=dtype)
            if cm.shape[0] < batch:
                cm = cm.repeat(batch, 1, 1)
                sl = sl.repeat(batch, 1, 1)
            c["chunk_masks"] = cm
            c["src_latents"] = sl
            c.update(extra_cond_fn(batch, device, dtype))
            if reference_latent is not None:
                _inject_reference_audio(c, reference_latent, batch, device, dtype)
            if not logged:
                logger.debug(f"[{log_tag}] Wrapper active: input={args['input'].shape} masks={cm.shape}")
                logged = True
            return apply_model(args["input"], args["timestep"], **c)

        self.model_patcher.set_model_unet_function_wrapper(model_function_wrapper)
        self._wrapper_applied = True

    def _get_sample_latents(self, noise):
        """Override to customize noise/latent_image passed to super().sample(). Returns (noise, latent_image)."""
        return noise, self.source_latent

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self._apply_model_wrapper()
        _apply_apg_guidance(self)
        sample_noise, sample_latent = self._get_sample_latents(noise)
        return super().sample(sample_noise, sample_latent, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)


# =============================================================================
# ACE-Step 1.5 Native Edit Guider (unified extend + repaint)
# =============================================================================

class ACEStep15NativeEditGuider(ACEStep15BaseGuider):
    """
    Unified ACE-Step 1.5 edit guider that handles both extend and repaint operations.

    This guider uses the model's native mask input mechanism, injecting chunk_masks
    and src_latents directly into the model's forward() method via model wrapping.

    Operations:
    - Extend: Add new content before (left) and/or after (right) the source audio
    - Repaint: Regenerate a specific time region within the audio

    Both operations can be combined in a single pass.

    The silence_latent is automatically loaded internally (downloaded from HuggingFace
    on first use). It's a learned tensor that tells the model to generate new content.
    """

    def __init__(self, model, positive, negative, cfg, source_latent,
                 extend_left_seconds=0.0, extend_right_seconds=0.0,
                 repaint_start_seconds=None, repaint_end_seconds=None,
                 reference_latent=None):
        super().__init__(model, positive, negative, cfg, source_latent, reference_latent)

        # Load silence_latent internally
        from .ace_step_utils import load_silence_latent
        silence_latent = load_silence_latent(verbose=True)

        # v1.5 frame rate: 48000 / 1920 = 25 fps
        self.fps = 25.0

        # Extension parameters
        self.extend_left_seconds = extend_left_seconds
        self.extend_right_seconds = extend_right_seconds
        self.left_frames = int(extend_left_seconds * self.fps)
        self.right_frames = int(extend_right_seconds * self.fps)

        batch_size, channels, source_length = source_latent.shape
        total_length = self.left_frames + source_length + self.right_frames

        # Repaint parameters (adjusted for extension offset)
        self.has_repaint = (repaint_start_seconds is not None and
                           repaint_end_seconds is not None and
                           repaint_end_seconds > repaint_start_seconds)

        if self.has_repaint:
            adjusted_start = repaint_start_seconds + extend_left_seconds
            adjusted_end = repaint_end_seconds + extend_left_seconds
            self.repaint_start_frame = int(adjusted_start * self.fps)
            self.repaint_end_frame = int(adjusted_end * self.fps)
            self.repaint_start_frame = max(0, min(self.repaint_start_frame, total_length))
            self.repaint_end_frame = max(0, min(self.repaint_end_frame, total_length))
        else:
            self.repaint_start_frame = 0
            self.repaint_end_frame = 0

        logger.debug(f"[ACE15_EDIT] Initializing")
        logger.debug(f"[ACE15_EDIT]   source_latent.shape: {source_latent.shape}")
        logger.debug(f"[ACE15_EDIT]   silence_latent.shape: {silence_latent.shape}")
        logger.debug(f"[ACE15_EDIT]   extend: left={extend_left_seconds}s ({self.left_frames}f), right={extend_right_seconds}s ({self.right_frames}f)")
        if self.has_repaint:
            logger.debug(f"[ACE15_EDIT]   repaint: {repaint_start_seconds}s-{repaint_end_seconds}s (frames {self.repaint_start_frame}-{self.repaint_end_frame} in extended space)")
        logger.debug(f"[ACE15_EDIT]   total_length: {total_length}")

        # Prepare silence_latent in ComfyUI format [D, total_length]
        silence_tiled = silence_latent[0, :total_length, :]
        if silence_tiled.shape[0] < total_length:
            num_tiles = (total_length // silence_tiled.shape[0]) + 1
            silence_tiled = silence_latent[0].repeat(num_tiles, 1)[:total_length, :]
        silence_tiled = silence_tiled.transpose(0, 1).to(
            device=source_latent.device, dtype=source_latent.dtype
        )

        # Create working latent
        if self.left_frames > 0 or self.right_frames > 0:
            self.working_latent = silence_tiled.unsqueeze(0).expand(batch_size, -1, -1).clone()
            self.working_latent[:, :, self.left_frames:self.left_frames + source_length] = source_latent
        else:
            self.working_latent = source_latent.clone()

        # Create chunk_masks (0 = preserve, 1 = generate)
        self.chunk_masks = torch.zeros_like(self.working_latent)
        if self.left_frames > 0:
            self.chunk_masks[:, :, :self.left_frames] = 1.0
        if self.right_frames > 0:
            self.chunk_masks[:, :, -self.right_frames:] = 1.0
        if self.has_repaint:
            self.chunk_masks[:, :, self.repaint_start_frame:self.repaint_end_frame] = 1.0

        # Create src_latents with silence_latent in generation regions
        self.src_latents = self.working_latent.clone()
        if self.has_repaint:
            silence_repaint = silence_tiled[:, self.repaint_start_frame:self.repaint_end_frame]
            self.src_latents[:, :, self.repaint_start_frame:self.repaint_end_frame] = silence_repaint.unsqueeze(0).expand(batch_size, -1, -1)

        # Log summary
        gen_frames = self.chunk_masks.sum().item() / (batch_size * channels)
        logger.debug(f"[ACE15_EDIT]   silence_tiled.shape: {silence_tiled.shape}")
        logger.debug(f"[ACE15_EDIT]   working_latent.shape: {self.working_latent.shape}")
        logger.debug(f"[ACE15_EDIT]   src_latents.shape: {self.src_latents.shape}")
        logger.debug(f"[ACE15_EDIT]   chunk_masks.shape: {self.chunk_masks.shape}")
        logger.debug(f"[ACE15_EDIT]   generation frames: {gen_frames:.0f} ({gen_frames / self.fps:.2f}s)")
        logger.debug(f"[ACE15_EDIT]   chunk_masks range: [{self.chunk_masks.min().item():.1f}, {self.chunk_masks.max().item():.1f}]")
        logger.debug(f"[ACE15_EDIT]   silence_latent reference: mean={silence_tiled.mean().item():.4f}, std={silence_tiled.std().item():.4f}")

        if self.left_frames > 0:
            logger.debug(f"[ACE15_EDIT]   left_extend working_latent: mean={self.working_latent[:, :, :self.left_frames].mean().item():.4f}")
            logger.debug(f"[ACE15_EDIT]   left_extend src_latents: mean={self.src_latents[:, :, :self.left_frames].mean().item():.4f}")
        if self.right_frames > 0:
            logger.debug(f"[ACE15_EDIT]   right_extend working_latent: mean={self.working_latent[:, :, -self.right_frames:].mean().item():.4f}")
            logger.debug(f"[ACE15_EDIT]   right_extend src_latents: mean={self.src_latents[:, :, -self.right_frames:].mean().item():.4f}")
        logger.debug(f"[ACE15_EDIT]   source region: mean={self.working_latent[:, :, self.left_frames:self.left_frames + source_length].mean().item():.4f}, std={self.working_latent[:, :, self.left_frames:self.left_frames + source_length].std().item():.4f}")

        if gen_frames == 0:
            logger.debug(f"[ACE15_EDIT]   WARNING: No generation frames! This will produce silence in extend regions.")

    def _get_wrapper_log_tag(self):
        return "ACE15_EDIT"

    def _get_extra_cond(self, input_batch_size, device, dtype):
        return {"is_covers": torch.zeros((input_batch_size,), device=device, dtype=torch.long)}

    def _get_sample_latents(self, noise):
        device = noise.device
        dtype = noise.dtype
        working_noise = torch.randn_like(self.working_latent, device=device, dtype=dtype)
        return working_noise, self.working_latent


# Backwards compatibility aliases
ACEStep15NativeExtendGuider = ACEStep15NativeEditGuider
ACEStep15NativeRepaintGuider = ACEStep15NativeEditGuider


# =============================================================================
# ACE-Step 1.5 Cover Guider
# =============================================================================

class ACEStep15NativeCoverGuider(ACEStep15BaseGuider):
    """
    ACE-Step 1.5 cover guider for style transfer/regeneration.

    Uses semantic tokens (lm_hints) from the source audio as structural guidance
    while generating new content with a different style/timbre.

    Official behavior:
    - chunk_masks: All 1s (generate everything)
    - is_covers: 1 (use semantic hints instead of raw latents)
    - precomputed_lm_hints_25Hz: Semantic representation of source audio

    If semantic_hints is not provided, they will be automatically extracted from
    the source_latent. For advanced workflows (e.g., blending hints from multiple
    sources), use ACEStep15SemanticExtractor and ACEStep15SemanticHintsBlend.
    """

    def __init__(self, model, positive, negative, cfg, source_latent, semantic_hints=None, reference_latent=None):
        super().__init__(model, positive, negative, cfg, source_latent, reference_latent)

        logger.debug(f"[ACE15_NATIVE_COVER] Initializing")
        logger.debug(f"[ACE15_NATIVE_COVER]   source_latent.shape: {source_latent.shape}")
        logger.debug(f"[ACE15_NATIVE_COVER]   source_latent stats: mean={source_latent.mean():.4f}, std={source_latent.std():.4f}")

        # Auto-extract semantic hints if not provided
        if semantic_hints is None:
            logger.debug(f"[ACE15_NATIVE_COVER]   No semantic_hints provided, auto-extracting...")
            from .ace_step_utils import extract_semantic_hints
            semantic_hints = extract_semantic_hints(model, source_latent, verbose=True)
            logger.debug(f"[ACE15_NATIVE_COVER]   Auto-extracted semantic_hints shape: {semantic_hints.shape}")

        self.semantic_hints = semantic_hints

        logger.debug(f"[ACE15_NATIVE_COVER]   semantic_hints stats: mean={semantic_hints.mean():.4f}, std={semantic_hints.std():.4f}")
        if semantic_hints.std() < 0.01:
            logger.debug(f"[ACE15_NATIVE_COVER]   WARNING: semantic_hints have very low variance ({semantic_hints.std():.6f}) - may be invalid!")
        if semantic_hints.shape != source_latent.shape:
            logger.debug(f"[ACE15_NATIVE_COVER]   WARNING: semantic_hints shape {semantic_hints.shape} != source_latent shape {source_latent.shape}")

        self.chunk_masks = torch.ones_like(source_latent)
        self.src_latents = source_latent.clone()

        logger.debug(f"[ACE15_NATIVE_COVER]   chunk_masks.shape: {self.chunk_masks.shape}")
        logger.debug(f"[ACE15_NATIVE_COVER]   mask sum: {self.chunk_masks.sum().item()} (all 1s)")

    def _get_wrapper_log_tag(self):
        return "ACE15_COVER"

    def _get_extra_cond(self, input_batch_size, device, dtype):
        extra = {}
        if self.semantic_hints is not None:
            sh = self.semantic_hints.to(device=device, dtype=dtype)
            if sh.shape[0] < input_batch_size:
                sh = sh.repeat(input_batch_size, 1, 1)
            extra["precomputed_lm_hints_25Hz"] = sh
            extra["is_covers"] = torch.ones((input_batch_size,), device=device, dtype=torch.long)
        else:
            extra["is_covers"] = torch.zeros((input_batch_size,), device=device, dtype=torch.long)
        return extra


# =============================================================================
# ACE-Step 1.5 Extract Guider
# =============================================================================

class ACEStep15NativeExtractGuider(ACEStep15BaseGuider):
    """
    ACE-Step 1.5 extract guider for extracting specific tracks.

    Extracts a specific track (e.g., vocals, drums, bass) from the source audio.
    The track_name is used in the text encoder instruction to guide extraction.

    Official behavior:
    - chunk_masks: All 1s (generate everything)
    - is_covers: 1 (use semantic hints for structure)
    - precomputed_lm_hints_25Hz: Semantic representation of source audio
    - Instruction: "Extract the {TRACK_NAME} track from the audio:"

    If semantic_hints is not provided, they will be automatically extracted from
    the source_latent. For advanced workflows, use ACEStep15SemanticExtractor.
    """

    def __init__(self, model, positive, negative, cfg, source_latent, track_name="vocals", semantic_hints=None, reference_latent=None):
        super().__init__(model, positive, negative, cfg, source_latent, reference_latent)

        self.track_name = track_name
        self.semantic_hints = semantic_hints

        logger.debug(f"[ACE15_NATIVE_EXTRACT] Initializing")
        logger.debug(f"[ACE15_NATIVE_EXTRACT]   source_latent.shape: {source_latent.shape}")
        logger.debug(f"[ACE15_NATIVE_EXTRACT]   track_name: {track_name}")

        self.chunk_masks = torch.ones_like(source_latent)
        self.src_latents = source_latent.clone()

        logger.debug(f"[ACE15_NATIVE_EXTRACT]   chunk_masks.shape: {self.chunk_masks.shape}")

    def _get_wrapper_log_tag(self):
        return "ACE15_EXTRACT"

    def _get_extra_cond(self, input_batch_size, device, dtype):
        extra = {"is_covers": torch.zeros((input_batch_size,), device=device, dtype=torch.long)}
        if self.semantic_hints is not None:
            sh = self.semantic_hints.to(device=device, dtype=dtype)
            if sh.shape[0] < input_batch_size:
                sh = sh.repeat(input_batch_size, 1, 1)
            extra["precomputed_lm_hints_25Hz"] = sh
        return extra


# =============================================================================
# ACE-Step 1.5 Lego Guider
# =============================================================================

class ACEStep15NativeLegoGuider(ACEStep15BaseGuider):
    """
    ACE-Step 1.5 lego guider for generating specific tracks within a region.

    Generates a specific track (e.g., vocals, drums) within a time region while
    preserving the rest. Useful for adding instruments to existing audio.

    Official behavior:
    - chunk_masks: 1s in generation region, 0s elsewhere
    - src_latents: Original audio with generation region replaced by silence_latent
    - Instruction: "Generate the {TRACK_NAME} track based on the audio context:"

    The silence_latent is automatically loaded internally (downloaded from HuggingFace
    on first use). It's a learned tensor that tells the model to generate new content.
    """

    def __init__(self, model, positive, negative, cfg, source_latent,
                 track_name="vocals", start_seconds=0.0, end_seconds=None, reference_latent=None):
        super().__init__(model, positive, negative, cfg, source_latent, reference_latent)

        self.track_name = track_name

        # Load silence_latent internally
        from .ace_step_utils import load_silence_latent
        silence_latent = load_silence_latent(verbose=True)
        self.start_seconds = start_seconds

        # v1.5 frame rate: 48000 / 1920 = 25 fps
        self.fps = 25.0

        batch_size, channels, total_length = source_latent.shape

        if end_seconds is None:
            end_seconds = total_length / self.fps
        self.end_seconds = end_seconds

        self.start_frame = int(start_seconds * self.fps)
        self.end_frame = int(end_seconds * self.fps)
        self.start_frame = max(0, min(self.start_frame, total_length))
        self.end_frame = max(0, min(self.end_frame, total_length))

        logger.debug(f"[ACE15_NATIVE_LEGO] Initializing")
        logger.debug(f"[ACE15_NATIVE_LEGO]   source_latent.shape: {source_latent.shape}")
        logger.debug(f"[ACE15_NATIVE_LEGO]   silence_latent.shape: {silence_latent.shape}")
        logger.debug(f"[ACE15_NATIVE_LEGO]   track_name: {track_name}")
        logger.debug(f"[ACE15_NATIVE_LEGO]   region: {start_seconds}s-{end_seconds}s (frames {self.start_frame}-{self.end_frame})")

        self.chunk_masks = torch.zeros_like(source_latent)
        self.chunk_masks[:, :, self.start_frame:self.end_frame] = 1.0

        # Tile silence_latent to at least total_length
        silence_tiled = silence_latent[0, :total_length, :]
        if silence_tiled.shape[0] < total_length:
            num_tiles = (total_length // silence_tiled.shape[0]) + 1
            silence_tiled = silence_latent[0].repeat(num_tiles, 1)[:total_length, :]
        silence_tiled = silence_tiled.transpose(0, 1)

        # Create src_latents by replacing lego region with silence
        self.src_latents = source_latent.clone()
        silence_region = silence_tiled[:, self.start_frame:self.end_frame].to(
            device=source_latent.device, dtype=source_latent.dtype
        )
        self.src_latents[:, :, self.start_frame:self.end_frame] = silence_region.unsqueeze(0).expand(batch_size, -1, -1)

        lego_frames = self.end_frame - self.start_frame
        logger.debug(f"[ACE15_NATIVE_LEGO]   chunk_masks.shape: {self.chunk_masks.shape}")
        logger.debug(f"[ACE15_NATIVE_LEGO]   lego frames: {lego_frames} ({lego_frames / self.fps:.2f}s)")
        logger.debug(f"[ACE15_NATIVE_LEGO]   src_latents lego region filled with silence_latent: frames {self.start_frame}-{self.end_frame}")

    def _get_wrapper_log_tag(self):
        return "ACE15_LEGO"

    def _get_extra_cond(self, input_batch_size, device, dtype):
        return {"is_covers": torch.zeros((input_batch_size,), device=device, dtype=torch.long)}

    def _get_sample_latents(self, noise):
        device = noise.device
        dtype = noise.dtype
        working_noise = torch.randn_like(self.source_latent, device=device, dtype=dtype)
        return working_noise, self.source_latent
