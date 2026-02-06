import torch
import comfy.samplers
import comfy.sample
import math
from .ace_step_utils import ACEStepLatentUtils

# NOTE: gigantic shoutout to the powerful acestep team https://github.com/ace-step/ACE-Step
# NOTE: And another massive shoutout.... the adapted code here is based on https://github.com/billwuhao/ComfyUI_ACE-Step

# NOTE: this implementation is experimental and beta with a minimum of testing


# =============================================================================
# ACE-Step 1.5 Native Edit Guider (unified extend + repaint)
# =============================================================================

class ACEStep15NativeEditGuider(comfy.samplers.CFGGuider):
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
                 repaint_start_seconds=None, repaint_end_seconds=None):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(cfg)

        self.source_latent = source_latent

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
            # Adjust repaint region for left extension offset
            adjusted_start = repaint_start_seconds + extend_left_seconds
            adjusted_end = repaint_end_seconds + extend_left_seconds
            self.repaint_start_frame = int(adjusted_start * self.fps)
            self.repaint_end_frame = int(adjusted_end * self.fps)
            # Clamp to valid range
            self.repaint_start_frame = max(0, min(self.repaint_start_frame, total_length))
            self.repaint_end_frame = max(0, min(self.repaint_end_frame, total_length))
        else:
            self.repaint_start_frame = 0
            self.repaint_end_frame = 0

        print(f"[ACE15_EDIT] Initializing")
        print(f"[ACE15_EDIT]   source_latent.shape: {source_latent.shape}")
        print(f"[ACE15_EDIT]   silence_latent.shape: {silence_latent.shape}")
        print(f"[ACE15_EDIT]   extend: left={extend_left_seconds}s ({self.left_frames}f), right={extend_right_seconds}s ({self.right_frames}f)")
        if self.has_repaint:
            print(f"[ACE15_EDIT]   repaint: {repaint_start_seconds}s-{repaint_end_seconds}s (frames {self.repaint_start_frame}-{self.repaint_end_frame} in extended space)")
        print(f"[ACE15_EDIT]   total_length: {total_length}")

        # Step 1: Prepare silence_latent in ComfyUI format [D, total_length]
        # silence_latent shape is [1, T, D], need [D, total_length] for ComfyUI format
        silence_tiled = silence_latent[0, :total_length, :]  # [total_length, D] or less
        if silence_tiled.shape[0] < total_length:
            # Need to tile/repeat
            num_tiles = (total_length // silence_tiled.shape[0]) + 1
            silence_tiled = silence_latent[0].repeat(num_tiles, 1)[:total_length, :]
        # Transpose to [D, total_length] to match latent channel-first format
        silence_tiled = silence_tiled.transpose(0, 1).to(
            device=source_latent.device, dtype=source_latent.dtype
        )  # [D, total_length]

        # Step 2: Create working latent
        # IMPORTANT: In the reference, zero audio → VAE encode → silence_latent
        # So extend regions should have silence_latent, not zeros
        if self.left_frames > 0 or self.right_frames > 0:
            # Initialize with silence_latent (matching reference behavior)
            self.working_latent = silence_tiled.unsqueeze(0).expand(batch_size, -1, -1).clone()
            # Copy source into the correct position (overwriting silence in source region)
            self.working_latent[:, :, self.left_frames:self.left_frames + source_length] = source_latent
        else:
            self.working_latent = source_latent.clone()

        # Step 3: Create chunk_masks (0 = preserve, 1 = generate)
        self.chunk_masks = torch.zeros_like(self.working_latent)

        # Mark extension regions for generation
        if self.left_frames > 0:
            self.chunk_masks[:, :, :self.left_frames] = 1.0
        if self.right_frames > 0:
            self.chunk_masks[:, :, -self.right_frames:] = 1.0

        # Mark repaint region for generation
        if self.has_repaint:
            self.chunk_masks[:, :, self.repaint_start_frame:self.repaint_end_frame] = 1.0

        # Step 4: Create src_latents with silence_latent in generation regions
        # Start from working_latent (which already has silence in extend regions)
        self.src_latents = self.working_latent.clone()

        # Fill repaint region with silence_latent (extend regions already have it from working_latent)
        # Use same indices as reference: src_latent[start:end] = silence_latent[start:end]
        if self.has_repaint:
            silence_repaint = silence_tiled[:, self.repaint_start_frame:self.repaint_end_frame]
            self.src_latents[:, :, self.repaint_start_frame:self.repaint_end_frame] = silence_repaint.unsqueeze(0).expand(batch_size, -1, -1)

        # Log summary
        gen_frames = self.chunk_masks.sum().item() / (batch_size * channels)
        print(f"[ACE15_EDIT]   silence_tiled.shape: {silence_tiled.shape}")
        print(f"[ACE15_EDIT]   working_latent.shape: {self.working_latent.shape}")
        print(f"[ACE15_EDIT]   src_latents.shape: {self.src_latents.shape}")
        print(f"[ACE15_EDIT]   chunk_masks.shape: {self.chunk_masks.shape}")
        print(f"[ACE15_EDIT]   generation frames: {gen_frames:.0f} ({gen_frames / self.fps:.2f}s)")

        # Verify chunk_masks has generation regions marked
        mask_max = self.chunk_masks.max().item()
        mask_min = self.chunk_masks.min().item()
        print(f"[ACE15_EDIT]   chunk_masks range: [{mask_min:.1f}, {mask_max:.1f}] (should have 1.0 for generation)")

        # Verify silence_latent was inserted (check mean/std of extend regions vs source region)
        silence_ref_mean = silence_tiled.mean().item()
        silence_ref_std = silence_tiled.std().item()
        print(f"[ACE15_EDIT]   silence_latent reference: mean={silence_ref_mean:.4f}, std={silence_ref_std:.4f}")

        if self.left_frames > 0:
            left_mean = self.working_latent[:, :, :self.left_frames].mean().item()
            left_std = self.working_latent[:, :, :self.left_frames].std().item()
            src_left_mean = self.src_latents[:, :, :self.left_frames].mean().item()
            print(f"[ACE15_EDIT]   left_extend working_latent: mean={left_mean:.4f}, std={left_std:.4f}")
            print(f"[ACE15_EDIT]   left_extend src_latents: mean={src_left_mean:.4f} (should match silence_latent)")
        if self.right_frames > 0:
            right_mean = self.working_latent[:, :, -self.right_frames:].mean().item()
            right_std = self.working_latent[:, :, -self.right_frames:].std().item()
            src_right_mean = self.src_latents[:, :, -self.right_frames:].mean().item()
            print(f"[ACE15_EDIT]   right_extend working_latent: mean={right_mean:.4f}, std={right_std:.4f}")
            print(f"[ACE15_EDIT]   right_extend src_latents: mean={src_right_mean:.4f} (should match silence_latent)")
        source_mean = self.working_latent[:, :, self.left_frames:self.left_frames + source_length].mean().item()
        source_std = self.working_latent[:, :, self.left_frames:self.left_frames + source_length].std().item()
        print(f"[ACE15_EDIT]   source region: mean={source_mean:.4f}, std={source_std:.4f}")

        # Warn if no generation regions
        if gen_frames == 0:
            print(f"[ACE15_EDIT]   WARNING: No generation frames! This will produce silence in extend regions.")

        self._wrapper_applied = False

    def _apply_model_wrapper(self):
        """Apply model wrapper to inject chunk_masks and src_latents"""
        if self._wrapper_applied:
            return

        chunk_masks = self.chunk_masks
        src_latents = self.src_latents

        # Clear any existing wrapper to avoid stale closure values
        self.model_patcher.set_model_unet_function_wrapper(None)

        def model_function_wrapper(apply_model, args):
            c = args["c"].copy()
            device = args["input"].device
            dtype = args["input"].dtype

            # Handle CFG batching
            input_batch_size = args["input"].shape[0]
            cm = chunk_masks.to(device=device, dtype=dtype)
            sl = src_latents.to(device=device, dtype=dtype)
            if cm.shape[0] < input_batch_size:
                cm = cm.repeat(input_batch_size, 1, 1)
                sl = sl.repeat(input_batch_size, 1, 1)

            c["chunk_masks"] = cm
            c["src_latents"] = sl
            # Explicitly set is_covers=0 for extend/repaint tasks
            # This tells the model to use src_latents directly (not semantic hints)
            c["is_covers"] = torch.zeros((input_batch_size,), device=device, dtype=torch.long)

            if not hasattr(model_function_wrapper, '_logged'):
                print(f"[ACE15_EDIT_WRAPPER] Injecting chunk_masks and src_latents")
                print(f"[ACE15_EDIT_WRAPPER]   input.shape: {args['input'].shape}")
                print(f"[ACE15_EDIT_WRAPPER]   chunk_masks.shape: {c['chunk_masks'].shape}")
                print(f"[ACE15_EDIT_WRAPPER]   src_latents.shape: {c['src_latents'].shape}")
                print(f"[ACE15_EDIT_WRAPPER]   is_covers: {c['is_covers']} (0=extend/repaint mode)")
                print(f"[ACE15_EDIT_WRAPPER]   chunk_masks sum: {cm.sum().item():.0f} (generation frames * batch * channels)")
                model_function_wrapper._logged = True

            return apply_model(args["input"], args["timestep"], **c)

        model_function_wrapper._logged = False
        self.model_patcher.set_model_unet_function_wrapper(model_function_wrapper)
        self._wrapper_applied = True
        print(f"[ACE15_EDIT] Model wrapper applied")

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to apply model wrapper and use working latent"""

        print(f"[ACE15_EDIT] sample() called")
        print(f"[ACE15_EDIT]   noise.shape: {noise.shape}")
        print(f"[ACE15_EDIT]   latent_image.shape: {latent_image.shape}")

        self._apply_model_wrapper()

        # Generate noise for working latent shape
        device = noise.device
        dtype = noise.dtype
        working_noise = torch.randn_like(self.working_latent, device=device, dtype=dtype)

        print(f"[ACE15_EDIT]   working_noise.shape: {working_noise.shape}")
        print(f"[ACE15_EDIT]   working_latent.shape: {self.working_latent.shape}")

        result = super().sample(
            working_noise,
            self.working_latent,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed
        )

        print(f"[ACE15_EDIT]   result.shape: {result.shape}")

        return result


# Backwards compatibility aliases
ACEStep15NativeExtendGuider = ACEStep15NativeEditGuider
ACEStep15NativeRepaintGuider = ACEStep15NativeEditGuider


class ACEStep15NativeCoverGuider(comfy.samplers.CFGGuider):
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

    def __init__(self, model, positive, negative, cfg, source_latent, semantic_hints=None):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(cfg)

        self.source_latent = source_latent

        batch_size, channels, total_length = source_latent.shape

        print(f"[ACE15_NATIVE_COVER] Initializing")
        print(f"[ACE15_NATIVE_COVER]   source_latent.shape: {source_latent.shape}")
        print(f"[ACE15_NATIVE_COVER]   source_latent stats: mean={source_latent.mean():.4f}, std={source_latent.std():.4f}")

        # Auto-extract semantic hints if not provided
        if semantic_hints is None:
            print(f"[ACE15_NATIVE_COVER]   No semantic_hints provided, auto-extracting...")
            from .ace_step_utils import extract_semantic_hints
            semantic_hints = extract_semantic_hints(model, source_latent, verbose=True)
            print(f"[ACE15_NATIVE_COVER]   Auto-extracted semantic_hints shape: {semantic_hints.shape}")

        self.semantic_hints = semantic_hints

        # Validate semantic hints
        print(f"[ACE15_NATIVE_COVER]   semantic_hints stats: mean={semantic_hints.mean():.4f}, std={semantic_hints.std():.4f}")
        if semantic_hints.std() < 0.01:
            print(f"[ACE15_NATIVE_COVER]   WARNING: semantic_hints have very low variance ({semantic_hints.std():.6f}) - may be invalid!")
        if semantic_hints.shape != source_latent.shape:
            print(f"[ACE15_NATIVE_COVER]   WARNING: semantic_hints shape {semantic_hints.shape} != source_latent shape {source_latent.shape}")

        # Cover: generate everything
        # chunk_masks: All 1s (generate all frames)
        self.chunk_masks = torch.ones_like(source_latent)

        # src_latents: Original audio (used as fallback if is_covers=0)
        self.src_latents = source_latent.clone()

        print(f"[ACE15_NATIVE_COVER]   chunk_masks.shape: {self.chunk_masks.shape}")
        print(f"[ACE15_NATIVE_COVER]   mask sum: {self.chunk_masks.sum().item()} (all 1s)")

        self._wrapper_applied = False

    def _apply_model_wrapper(self):
        """Apply model wrapper to inject chunk_masks, src_latents, is_covers, and semantic hints"""
        if self._wrapper_applied:
            return

        chunk_masks = self.chunk_masks
        src_latents = self.src_latents
        semantic_hints = self.semantic_hints

        self.model_patcher.set_model_unet_function_wrapper(None)

        def model_function_wrapper(apply_model, args):
            c = args["c"].copy()
            device = args["input"].device
            dtype = args["input"].dtype

            input_batch_size = args["input"].shape[0]
            cm = chunk_masks.to(device=device, dtype=dtype)
            sl = src_latents.to(device=device, dtype=dtype)
            if cm.shape[0] < input_batch_size:
                cm = cm.repeat(input_batch_size, 1, 1)
                sl = sl.repeat(input_batch_size, 1, 1)

            c["chunk_masks"] = cm
            c["src_latents"] = sl

            # Set is_covers based on whether we have semantic hints
            if semantic_hints is not None:
                # Proper cover: use semantic hints via is_covers=1
                sh = semantic_hints.to(device=device, dtype=dtype)
                if sh.shape[0] < input_batch_size:
                    sh = sh.repeat(input_batch_size, 1, 1)
                c["precomputed_lm_hints_25Hz"] = sh
                c["is_covers"] = torch.ones((input_batch_size,), device=device, dtype=torch.long)
            else:
                # Fallback: use raw VAE latents via is_covers=0
                c["is_covers"] = torch.zeros((input_batch_size,), device=device, dtype=torch.long)

            if not hasattr(model_function_wrapper, '_logged'):
                print(f"[ACE15_COVER_WRAPPER] Injecting cover parameters")
                print(f"[ACE15_COVER_WRAPPER]   input.shape: {args['input'].shape}")
                print(f"[ACE15_COVER_WRAPPER]   input stats: mean={args['input'].mean():.4f}, std={args['input'].std():.4f}")
                print(f"[ACE15_COVER_WRAPPER]   chunk_masks.shape: {c['chunk_masks'].shape}")
                print(f"[ACE15_COVER_WRAPPER]   chunk_masks sum: {c['chunk_masks'].sum().item()} (should be all 1s for cover)")
                print(f"[ACE15_COVER_WRAPPER]   src_latents.shape: {c['src_latents'].shape}")
                print(f"[ACE15_COVER_WRAPPER]   src_latents stats: mean={c['src_latents'].mean():.4f}, std={c['src_latents'].std():.4f}")
                print(f"[ACE15_COVER_WRAPPER]   is_covers: {c['is_covers']} (1=use semantic hints, 0=use VAE latents)")
                if "precomputed_lm_hints_25Hz" in c:
                    sh = c["precomputed_lm_hints_25Hz"]
                    print(f"[ACE15_COVER_WRAPPER]   precomputed_lm_hints_25Hz.shape: {sh.shape}")
                    print(f"[ACE15_COVER_WRAPPER]   precomputed_lm_hints_25Hz stats: mean={sh.mean():.4f}, std={sh.std():.4f}, min={sh.min():.4f}, max={sh.max():.4f}")
                    # Check if hints look valid (should have some variance, not constant)
                    if sh.std() < 0.01:
                        print(f"[ACE15_COVER_WRAPPER]   WARNING: semantic hints have very low variance! May be constant/invalid.")
                else:
                    print(f"[ACE15_COVER_WRAPPER]   WARNING: No precomputed_lm_hints_25Hz - will fall back to VAE latents!")
                model_function_wrapper._logged = True

            return apply_model(args["input"], args["timestep"], **c)

        model_function_wrapper._logged = False
        self.model_patcher.set_model_unet_function_wrapper(model_function_wrapper)
        self._wrapper_applied = True
        print(f"[ACE15_NATIVE_COVER] Model wrapper applied")

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to apply model wrapper for cover"""

        print(f"[ACE15_NATIVE_COVER] sample() called")
        print(f"[ACE15_NATIVE_COVER]   noise.shape: {noise.shape}")
        print(f"[ACE15_NATIVE_COVER]   latent_image.shape: {latent_image.shape}")

        self._apply_model_wrapper()

        device = noise.device
        dtype = noise.dtype
        working_noise = torch.randn_like(self.source_latent, device=device, dtype=dtype)

        result = super().sample(
            working_noise,
            self.source_latent,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed
        )

        print(f"[ACE15_NATIVE_COVER]   result.shape: {result.shape}")

        return result


class ACEStep15NativeExtractGuider(comfy.samplers.CFGGuider):
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

    def __init__(self, model, positive, negative, cfg, source_latent, track_name="vocals", semantic_hints=None):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(cfg)

        self.source_latent = source_latent
        self.track_name = track_name

        batch_size, channels, total_length = source_latent.shape

        print(f"[ACE15_NATIVE_EXTRACT] Initializing")
        print(f"[ACE15_NATIVE_EXTRACT]   source_latent.shape: {source_latent.shape}")
        print(f"[ACE15_NATIVE_EXTRACT]   track_name: {track_name}")

        # Auto-extract semantic hints if not provided
        if semantic_hints is None:
            print(f"[ACE15_NATIVE_EXTRACT]   No semantic_hints provided, auto-extracting...")
            from .ace_step_utils import extract_semantic_hints
            semantic_hints = extract_semantic_hints(model, source_latent, verbose=True)
            print(f"[ACE15_NATIVE_EXTRACT]   Auto-extracted semantic_hints shape: {semantic_hints.shape}")

        self.semantic_hints = semantic_hints

        # Validate semantic hints
        print(f"[ACE15_NATIVE_EXTRACT]   semantic_hints stats: mean={semantic_hints.mean():.4f}, std={semantic_hints.std():.4f}")

        # Extract: generate everything (the extracted track)
        self.chunk_masks = torch.ones_like(source_latent)
        self.src_latents = source_latent.clone()

        print(f"[ACE15_NATIVE_EXTRACT]   chunk_masks.shape: {self.chunk_masks.shape}")

        self._wrapper_applied = False

    def _apply_model_wrapper(self):
        """Apply model wrapper to inject chunk_masks, src_latents, is_covers, and semantic hints"""
        if self._wrapper_applied:
            return

        chunk_masks = self.chunk_masks
        src_latents = self.src_latents
        semantic_hints = self.semantic_hints

        self.model_patcher.set_model_unet_function_wrapper(None)

        def model_function_wrapper(apply_model, args):
            c = args["c"].copy()
            device = args["input"].device
            dtype = args["input"].dtype

            input_batch_size = args["input"].shape[0]
            cm = chunk_masks.to(device=device, dtype=dtype)
            sl = src_latents.to(device=device, dtype=dtype)
            if cm.shape[0] < input_batch_size:
                cm = cm.repeat(input_batch_size, 1, 1)
                sl = sl.repeat(input_batch_size, 1, 1)

            c["chunk_masks"] = cm
            c["src_latents"] = sl

            # Set is_covers based on whether we have semantic hints
            if semantic_hints is not None:
                sh = semantic_hints.to(device=device, dtype=dtype)
                if sh.shape[0] < input_batch_size:
                    sh = sh.repeat(input_batch_size, 1, 1)
                c["precomputed_lm_hints_25Hz"] = sh
                c["is_covers"] = torch.ones((input_batch_size,), device=device, dtype=torch.long)
            else:
                c["is_covers"] = torch.zeros((input_batch_size,), device=device, dtype=torch.long)

            if not hasattr(model_function_wrapper, '_logged'):
                print(f"[ACE15_EXTRACT_WRAPPER] Injecting extract parameters")
                print(f"[ACE15_EXTRACT_WRAPPER]   is_covers: {c['is_covers']}")
                if "precomputed_lm_hints_25Hz" in c:
                    print(f"[ACE15_EXTRACT_WRAPPER]   precomputed_lm_hints_25Hz.shape: {c['precomputed_lm_hints_25Hz'].shape}")
                model_function_wrapper._logged = True

            return apply_model(args["input"], args["timestep"], **c)

        model_function_wrapper._logged = False
        self.model_patcher.set_model_unet_function_wrapper(model_function_wrapper)
        self._wrapper_applied = True
        print(f"[ACE15_NATIVE_EXTRACT] Model wrapper applied")

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to apply model wrapper for extract"""

        print(f"[ACE15_NATIVE_EXTRACT] sample() called")

        self._apply_model_wrapper()

        device = noise.device
        dtype = noise.dtype
        working_noise = torch.randn_like(self.source_latent, device=device, dtype=dtype)

        result = super().sample(
            working_noise,
            self.source_latent,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed
        )

        print(f"[ACE15_NATIVE_EXTRACT]   result.shape: {result.shape}")

        return result


class ACEStep15NativeLegoGuider(comfy.samplers.CFGGuider):
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
                 track_name="vocals", start_seconds=0.0, end_seconds=None):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(cfg)

        self.source_latent = source_latent
        self.track_name = track_name

        # Load silence_latent internally
        from .ace_step_utils import load_silence_latent
        silence_latent = load_silence_latent(verbose=True)
        self.start_seconds = start_seconds

        # v1.5 frame rate: 48000 / 1920 = 25 fps
        self.fps = 25.0

        batch_size, channels, total_length = source_latent.shape

        # Calculate end time if not provided (use full duration)
        if end_seconds is None:
            end_seconds = total_length / self.fps
        self.end_seconds = end_seconds

        self.start_frame = int(start_seconds * self.fps)
        self.end_frame = int(end_seconds * self.fps)

        # Clamp frame indices to valid range
        self.start_frame = max(0, min(self.start_frame, total_length))
        self.end_frame = max(0, min(self.end_frame, total_length))

        print(f"[ACE15_NATIVE_LEGO] Initializing")
        print(f"[ACE15_NATIVE_LEGO]   source_latent.shape: {source_latent.shape}")
        print(f"[ACE15_NATIVE_LEGO]   silence_latent.shape: {silence_latent.shape}")
        print(f"[ACE15_NATIVE_LEGO]   track_name: {track_name}")
        print(f"[ACE15_NATIVE_LEGO]   region: {start_seconds}s-{end_seconds}s (frames {self.start_frame}-{self.end_frame})")

        # Lego: generate track in region, preserve elsewhere
        self.chunk_masks = torch.zeros_like(source_latent)
        self.chunk_masks[:, :, self.start_frame:self.end_frame] = 1.0

        # src_latents: Original audio with generation region replaced by silence_latent
        # Same logic as repaint - silence_latent shape is [1, T, D], need [batch, D, T]
        silence_length = silence_latent.shape[1]  # T dimension
        lego_length = self.end_frame - self.start_frame

        # Tile silence_latent to at least total_length
        silence_tiled = silence_latent[0, :total_length, :]  # [total_length, D] or less
        if silence_tiled.shape[0] < total_length:
            # Need to tile/repeat
            num_tiles = (total_length // silence_tiled.shape[0]) + 1
            silence_tiled = silence_latent[0].repeat(num_tiles, 1)[:total_length, :]

        # Transpose to [D, total_length] to match latent channel-first format
        silence_tiled = silence_tiled.transpose(0, 1)  # [D, total_length]

        # Create src_latents by replacing lego region with silence
        self.src_latents = source_latent.clone()
        # Move silence to same device/dtype as source
        silence_region = silence_tiled[:, self.start_frame:self.end_frame].to(
            device=source_latent.device, dtype=source_latent.dtype
        )
        self.src_latents[:, :, self.start_frame:self.end_frame] = silence_region.unsqueeze(0).expand(batch_size, -1, -1)

        lego_frames = self.end_frame - self.start_frame
        print(f"[ACE15_NATIVE_LEGO]   chunk_masks.shape: {self.chunk_masks.shape}")
        print(f"[ACE15_NATIVE_LEGO]   lego frames: {lego_frames} ({lego_frames / self.fps:.2f}s)")
        print(f"[ACE15_NATIVE_LEGO]   src_latents lego region filled with silence_latent: frames {self.start_frame}-{self.end_frame}")

        self._wrapper_applied = False

    def _apply_model_wrapper(self):
        """Apply model wrapper to inject chunk_masks and src_latents"""
        if self._wrapper_applied:
            return

        chunk_masks = self.chunk_masks
        src_latents = self.src_latents

        self.model_patcher.set_model_unet_function_wrapper(None)

        def model_function_wrapper(apply_model, args):
            c = args["c"].copy()
            device = args["input"].device
            dtype = args["input"].dtype

            input_batch_size = args["input"].shape[0]
            cm = chunk_masks.to(device=device, dtype=dtype)
            sl = src_latents.to(device=device, dtype=dtype)
            if cm.shape[0] < input_batch_size:
                cm = cm.repeat(input_batch_size, 1, 1)
                sl = sl.repeat(input_batch_size, 1, 1)

            c["chunk_masks"] = cm
            c["src_latents"] = sl

            if not hasattr(model_function_wrapper, '_logged'):
                print(f"[ACE15_LEGO_WRAPPER] Injecting chunk_masks and src_latents")
                model_function_wrapper._logged = True

            return apply_model(args["input"], args["timestep"], **c)

        model_function_wrapper._logged = False
        self.model_patcher.set_model_unet_function_wrapper(model_function_wrapper)
        self._wrapper_applied = True
        print(f"[ACE15_NATIVE_LEGO] Model wrapper applied")

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to apply model wrapper for lego"""

        print(f"[ACE15_NATIVE_LEGO] sample() called")

        self._apply_model_wrapper()

        device = noise.device
        dtype = noise.dtype
        working_noise = torch.randn_like(self.source_latent, device=device, dtype=dtype)

        result = super().sample(
            working_noise,
            self.source_latent,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed
        )

        print(f"[ACE15_NATIVE_LEGO]   result.shape: {result.shape}")

        return result


# =============================================================================
# ACE-Step 1.0 Guiders (use latent-level ODE blending)
# =============================================================================

class ACEStepRepaintGuider(comfy.samplers.CFGGuider):
    """Repaint guider implementation"""

    def __init__(self, model, positive, negative, cfg, source_latent,
                 start_time, end_time, repaint_strength, feather_time):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(cfg)

        self.source_latent = source_latent
        self.start_time = start_time
        self.end_time = end_time
        self.repaint_strength = repaint_strength

        # Detect version from latent shape
        self.version = ACEStepLatentUtils.detect_version(source_latent)
        self.is_v1_5 = self.version == ACEStepLatentUtils.V1_5

        # Convert times to frame indices
        self.start_frame = ACEStepLatentUtils.time_to_frame_index(start_time, self.version)
        self.end_frame = ACEStepLatentUtils.time_to_frame_index(end_time, self.version)

        # Create binary repaint mask (1.0 = generate new, 0.0 = preserve original)
        self.repaint_mask = torch.zeros_like(source_latent)
        if self.is_v1_5:
            self.repaint_mask[:, :, self.start_frame:self.end_frame] = 1.0
        else:
            self.repaint_mask[:, :, :, self.start_frame:self.end_frame] = 1.0

        # Apply feathering if requested
        if feather_time > 0:
            feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_time, self.version)
            self.repaint_mask = ACEStepLatentUtils.create_feather_mask(
                self.repaint_mask, feather_frames)

        # Calculate n_min based on repaint_strength
        self.retake_variance = repaint_strength
    
    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to implement repaint logic"""

        print(f"[REPAINT_GUIDER] sample() called")
        print(f"[REPAINT_GUIDER]   noise.shape: {noise.shape}")
        print(f"[REPAINT_GUIDER]   latent_image.shape: {latent_image.shape}")
        print(f"[REPAINT_GUIDER]   self.is_v1_5: {self.is_v1_5}")
        print(f"[REPAINT_GUIDER]   self.version: {self.version}")
        print(f"[REPAINT_GUIDER]   self.source_latent.shape: {self.source_latent.shape}")
        print(f"[REPAINT_GUIDER]   self.repaint_mask.shape: {self.repaint_mask.shape}")

        # Calculate n_min
        total_steps = len(sigmas) - 1
        n_min = int(total_steps * (1 - self.retake_variance))
        
        # Core variables
        x0 = self.source_latent  # ground truth latents
        z0 = noise               # initial noise (FIXED, not regenerated each step)
        zt_edit = x0.clone()     # will be updated
        is_repaint = True
        repaint_mask = self.repaint_mask
        
        # Convert sigmas to timesteps
        timesteps = (sigmas * 1000).long()
        
        # Store original dtype for precision
        original_dtype = noise.dtype
        
        # Initialize target_latents
        target_latents = noise.clone()
        
        def repaint_callback(step, x0_unused, x, total_steps):
            """Implement repaint logic"""
            nonlocal target_latents, zt_edit
            
            device = x.device
            i = step
            t = timesteps[step] if step < len(timesteps) else timesteps[-1]
            
            # Repaint initialization
            if is_repaint:
                if i < n_min:
                    # Let normal scheduler work for steps before n_min
                    target_latents = x.clone()
                    return
                elif i == n_min:
                    t_i = t.float() / 1000.0
                    zt_src = (1 - t_i) * x0.to(device) + t_i * z0.to(device)
                    target_latents = zt_edit.to(device) + zt_src - x0.to(device)
                    
                    # Replace x with our initialized target_latents
                    x[:] = target_latents
                    return
            
            # For steps after n_min: BYPASS SCHEDULER and do custom ODE
            if is_repaint and i >= n_min and i > 0:
                # Get timesteps for ODE integration
                t_i = t.float() / 1000.0
                if i + 1 < len(timesteps):
                    t_im1 = timesteps[i + 1].float() / 1000.0
                else:
                    t_im1 = torch.zeros_like(t_i).to(t_i.device)
                
                # Dtype conversion for precision
                target_latents = target_latents.to(torch.float32)
                
                # Apply custom ODE step
                # Note: The scheduler has already done its step, so x contains the "prev_sample"
                # We'll use this as our ODE result for the generated regions
                prev_sample = x.to(torch.float32)
                
                # Convert back to original dtype
                prev_sample = prev_sample.to(original_dtype)
                target_latents = prev_sample

                zt_src = (1 - t_im1) * x0.to(device) + t_im1 * z0.to(device)
                repaint_mask_device = repaint_mask.to(device)
                target_latents = torch.where(repaint_mask_device == 1.0, target_latents, zt_src)
                
                # Replace x with our custom ODE result
                x[:] = target_latents
            
            # Call original callback if provided
            if callback is not None:
                return callback(step, x0_unused, x, total_steps)
        
        # Run sampling with repaint callback
        result = super().sample(noise, latent_image, sampler, sigmas, denoise_mask, repaint_callback, disable_pbar, seed)
        
        return result

class ACEStepExtendGuider(comfy.samplers.CFGGuider):
    """Extend guider implementation with custom ODE"""

    def __init__(self, model, positive, negative, cfg, source_latent,
                 extend_left_time, extend_right_time):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(cfg)

        self.source_latent = source_latent
        self.extend_left_time = extend_left_time
        self.extend_right_time = extend_right_time

        # Detect version from latent shape
        self.version = ACEStepLatentUtils.detect_version(source_latent)
        self.is_v1_5 = self.version == ACEStepLatentUtils.V1_5

        # Calculate frame extensions using version-specific FPS
        self.left_frames = ACEStepLatentUtils.time_to_frame_index(extend_left_time, self.version)
        self.right_frames = ACEStepLatentUtils.time_to_frame_index(extend_right_time, self.version)

        # Store original source length for post-processing
        self.src_latents_length = source_latent.shape[-1]

        # Calculate max inference length (240 seconds)
        fps = ACEStepLatentUtils.get_fps(self.version)
        self.max_infer_frame_length = int(240 * fps)
        
        # Initialize trimming variables (matching community logic)
        self.left_trim_length = 0
        self.right_trim_length = 0
        self.to_left_pad_gt_latents = None
        self.to_right_pad_gt_latents = None

        # Create extended latent with trimming logic (matching community exactly)
        gt_latents = source_latent
        frame_length = self.left_frames + gt_latents.shape[-1] + self.right_frames

        # Padding format differs: v1.5 is 3D (last dim), v1.0 is 4D (last dim)
        # torch.nn.functional.pad pads last dimensions, so (left, right) works for both

        # Handle left extension with trimming
        if self.left_frames > 0:
            extend_gt_latents = torch.nn.functional.pad(
                gt_latents, (self.left_frames, 0), "constant", 0
            )
            if frame_length > self.max_infer_frame_length:
                self.right_trim_length = frame_length - self.max_infer_frame_length
                if self.is_v1_5:
                    self.to_right_pad_gt_latents = extend_gt_latents[:, :, -self.right_trim_length:]
                    extend_gt_latents = extend_gt_latents[:, :, :self.max_infer_frame_length]
                else:
                    self.to_right_pad_gt_latents = extend_gt_latents[:, :, :, -self.right_trim_length:]
                    extend_gt_latents = extend_gt_latents[:, :, :, :self.max_infer_frame_length]
                frame_length = self.max_infer_frame_length
            gt_latents = extend_gt_latents

        # Handle right extension with trimming
        if self.right_frames > 0:
            extend_gt_latents = torch.nn.functional.pad(
                gt_latents, (0, self.right_frames), "constant", 0
            )
            frame_length = extend_gt_latents.shape[-1]
            if frame_length > self.max_infer_frame_length:
                self.left_trim_length = frame_length - self.max_infer_frame_length
                if self.is_v1_5:
                    self.to_left_pad_gt_latents = extend_gt_latents[:, :, :self.left_trim_length]
                    extend_gt_latents = extend_gt_latents[:, :, -self.max_infer_frame_length:]
                else:
                    self.to_left_pad_gt_latents = extend_gt_latents[:, :, :, :self.left_trim_length]
                    extend_gt_latents = extend_gt_latents[:, :, :, -self.max_infer_frame_length:]
                frame_length = self.max_infer_frame_length
            gt_latents = extend_gt_latents

        # Store the working latent (this is what we'll use for inference)
        self.extended_latent = gt_latents

        # Create binary mask for extended regions
        self.extend_mask = torch.zeros_like(self.extended_latent)
        if self.left_frames > 0:
            actual_left_frames = min(self.left_frames, self.extended_latent.shape[-1])
            if self.is_v1_5:
                self.extend_mask[:, :, :actual_left_frames] = 1.0
            else:
                self.extend_mask[:, :, :, :actual_left_frames] = 1.0
        if self.right_frames > 0:
            actual_right_frames = min(self.right_frames, self.extended_latent.shape[-1])
            if self.is_v1_5:
                self.extend_mask[:, :, -actual_right_frames:] = 1.0
            else:
                self.extend_mask[:, :, :, -actual_right_frames:] = 1.0

        # Calculate n_min based on retake_variance (using 1.0 for extend)
        self.retake_variance = 1.0
        
    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to implement extend logic exactly matching community implementation"""

        print(f"[EXTEND_GUIDER] sample() called")
        print(f"[EXTEND_GUIDER]   noise.shape: {noise.shape}")
        print(f"[EXTEND_GUIDER]   latent_image.shape: {latent_image.shape}")
        print(f"[EXTEND_GUIDER]   self.is_v1_5: {self.is_v1_5}")
        print(f"[EXTEND_GUIDER]   self.version: {self.version}")
        print(f"[EXTEND_GUIDER]   self.extended_latent.shape: {self.extended_latent.shape}")
        print(f"[EXTEND_GUIDER]   self.source_latent.shape: {self.source_latent.shape}")
        print(f"[EXTEND_GUIDER]   self.left_frames: {self.left_frames}")
        print(f"[EXTEND_GUIDER]   self.right_frames: {self.right_frames}")
        print(f"[EXTEND_GUIDER]   self.left_trim_length: {self.left_trim_length}")
        print(f"[EXTEND_GUIDER]   self.right_trim_length: {self.right_trim_length}")

        device = noise.device
        dtype = noise.dtype

        # Generate retake_latents - separate noise for extended regions (matches community)
        # Community uses randn_tensor with retake_random_generators, we approximate with randn_like
        retake_latents = torch.randn_like(self.extended_latent, device=device, dtype=dtype)
        print(f"[EXTEND_GUIDER]   retake_latents.shape: {retake_latents.shape}")

        # Generate target_latents for the FULL extended duration (this is what community does)
        # Community generates target_latents for the full frame_length, then slices it
        target_latents_full = torch.randn_like(self.extended_latent, device=device, dtype=dtype)
        print(f"[EXTEND_GUIDER]   target_latents_full.shape: {target_latents_full.shape}")

        # Create target_latents by concatenating pieces EXACTLY like community implementation
        # Community code lines 768-775: proper slicing with trimming
        padd_list = []

        # Left extension: use retake_latents directly
        if self.left_frames > 0:
            actual_left_frames = min(self.left_frames, retake_latents.shape[-1])
            print(f"[EXTEND_GUIDER]   LEFT: actual_left_frames={actual_left_frames}")
            # v1.5 is 3D, v1.0 is 4D - use appropriate indexing
            if self.is_v1_5:
                padd_list.append(retake_latents[:, :, :actual_left_frames])
            else:
                padd_list.append(retake_latents[:, :, :, :actual_left_frames])

        # Middle section: use TRIMMED SLICE of target_latents_full (this is the key fix!)
        # Community: target_latents[left_trim_length : target_latents.shape[-1] - right_trim_length]
        middle_start = self.left_trim_length
        middle_end = target_latents_full.shape[-1] - self.right_trim_length
        print(f"[EXTEND_GUIDER]   MIDDLE: middle_start={middle_start}, middle_end={middle_end}")

        # v1.5 is 3D, v1.0 is 4D - use appropriate indexing
        if self.is_v1_5:
            middle_slice = target_latents_full[:, :, middle_start:middle_end]
        else:
            middle_slice = target_latents_full[:, :, :, middle_start:middle_end]
        print(f"[EXTEND_GUIDER]   middle_slice.shape after slicing: {middle_slice.shape}")

        # The middle slice should match the original source latent length
        # We need to trim it to match the source latent size
        if middle_slice.shape[-1] > self.src_latents_length:
            # Trim to source length
            print(f"[EXTEND_GUIDER]   Trimming middle_slice from {middle_slice.shape[-1]} to {self.src_latents_length}")
            if self.is_v1_5:
                middle_slice = middle_slice[:, :, :self.src_latents_length]
            else:
                middle_slice = middle_slice[:, :, :, :self.src_latents_length]
        elif middle_slice.shape[-1] < self.src_latents_length:
            # This shouldn't happen in normal cases, but pad if needed
            padding_needed = self.src_latents_length - middle_slice.shape[-1]
            print(f"[EXTEND_GUIDER]   Padding middle_slice by {padding_needed}")
            middle_slice = torch.nn.functional.pad(middle_slice, (0, padding_needed), "constant", 0.0)

        print(f"[EXTEND_GUIDER]   middle_slice.shape final: {middle_slice.shape}")
        padd_list.append(middle_slice)

        # Right extension: use retake_latents directly
        if self.right_frames > 0:
            actual_right_frames = min(self.right_frames, retake_latents.shape[-1])
            print(f"[EXTEND_GUIDER]   RIGHT: actual_right_frames={actual_right_frames}")
            # v1.5 is 3D, v1.0 is 4D - use appropriate indexing
            if self.is_v1_5:
                padd_list.append(retake_latents[:, :, -actual_right_frames:])
            else:
                padd_list.append(retake_latents[:, :, :, -actual_right_frames:])
        
        target_latents = torch.cat(padd_list, dim=-1)
        
        # Ensure target_latents matches extended_latent shape
        assert target_latents.shape[-1] == self.extended_latent.shape[-1], \
            f"Shape mismatch: {target_latents.shape=} vs {self.extended_latent.shape=}"
        
        # Calculate n_min
        total_steps = len(sigmas) - 1
        n_min = int(total_steps * (1 - self.retake_variance))
        
        # Core variables - exactly matching community variable names and assignments
        x0 = self.extended_latent  # gt_latents in community
        z0 = target_latents        # target_latents in community (becomes z0)
        zt_edit = x0.clone()       # zt_edit = x0.clone() in community
        is_repaint = True          # extend is treated as repaint in community
        repaint_mask = self.extend_mask  # repaint_mask in community
        
        # Convert sigmas to timesteps
        timesteps = (sigmas * 1000).long()
        
        # Store original dtype for precision
        original_dtype = target_latents.dtype
        
        def extend_callback(step, x0_unused, x, total_steps):
            """Implement extend logic exactly matching community ODE"""
            nonlocal target_latents, zt_edit
            
            device = x.device
            i = step
            t = timesteps[step] if step < len(timesteps) else timesteps[-1]
            
            # Community repaint initialization logic
            if is_repaint:
                if i < n_min:
                    # Let normal scheduler work for steps before n_min (community: continue)
                    target_latents = x.clone()
                    return
                elif i == n_min:
                    # Community initialization at n_min
                    t_i = t.float() / 1000.0
                    zt_src = (1 - t_i) * x0.to(device) + t_i * z0.to(device)
                    target_latents = zt_edit.to(device) + zt_src - x0.to(device)
                    
                    # Replace x with our initialized target_latents
                    x[:] = target_latents
                    return
            
            # Community ODE step for i >= n_min
            if is_repaint and i >= n_min and i > 0:
                # Get timesteps for ODE integration (matches community exactly)
                t_i = t.float() / 1000.0
                if i + 1 < len(timesteps):
                    t_im1 = timesteps[i + 1].float() / 1000.0
                else:
                    t_im1 = torch.zeros_like(t_i).to(t_i.device)
                
                # Community ODE step: target_latents = target_latents + (t_im1 - t_i) * noise_pred
                # But we're in callback after scheduler step, so x already contains the result
                # We use x as our prev_sample (matches community's prev_sample)
                dtype = x.dtype  # Use x's dtype (matches community)
                target_latents = target_latents.to(torch.float32)
                prev_sample = x.to(torch.float32)
                prev_sample = prev_sample.to(dtype)
                target_latents = prev_sample
                
                # Community masking step: preserve source regions, keep generated in extended regions
                zt_src = (1 - t_im1) * x0.to(device) + t_im1 * z0.to(device)
                repaint_mask_device = repaint_mask.to(device)
                target_latents = torch.where(repaint_mask_device == 1.0, target_latents, zt_src)
                
                # Replace x with our result
                x[:] = target_latents
            
            # Call original callback if provided
            if callback is not None:
                return callback(step, x0_unused, x, total_steps)
        
        # Use extended latent and target_latents as initial noise
        result = super().sample(target_latents, self.extended_latent, sampler, sigmas, denoise_mask, extend_callback, disable_pbar, seed)
        
        # POST-PROCESSING: Apply community's final concatenation logic (lines 1140-1147)
        # This is the critical missing piece that makes extend sound like continuation!
        if self.to_right_pad_gt_latents is not None:
            result = torch.cat([result, self.to_right_pad_gt_latents.to(result.device)], dim=-1)
        if self.to_left_pad_gt_latents is not None:
            # Fix the bug in community code: should be [to_left_pad_gt_latents, result] not [to_right_pad_gt_latents, result]
            result = torch.cat([self.to_left_pad_gt_latents.to(result.device), result], dim=-1)
        
        return result

class ACEStepHybridGuider(comfy.samplers.CFGGuider):
    """Simple guider that combines repaint and extend functionality using latent replacement"""

    def __init__(self, model, positive, negative, cfg, source_latent,
                 repaint_start_time, repaint_end_time, repaint_strength, feather_time,
                 extend_left_time, extend_right_time):
        super().__init__(model)
        self.set_conds(positive, negative)
        self.set_cfg(cfg)

        self.source_latent = source_latent
        self.repaint_start_time = repaint_start_time
        self.repaint_end_time = repaint_end_time
        self.repaint_strength = repaint_strength
        self.feather_time = feather_time
        self.extend_left_time = extend_left_time
        self.extend_right_time = extend_right_time

        # Detect version from latent shape
        self.version = ACEStepLatentUtils.detect_version(source_latent)
        self.is_v1_5 = self.version == ACEStepLatentUtils.V1_5

        # Check which operations are enabled
        self.has_repaint = repaint_start_time is not None and repaint_end_time is not None
        self.has_extend = extend_left_time > 0 or extend_right_time > 0

        # Setup extend if enabled (using improved logic from ACEStepExtendGuider)
        if self.has_extend:
            self.left_frames = ACEStepLatentUtils.time_to_frame_index(extend_left_time, self.version)
            self.right_frames = ACEStepLatentUtils.time_to_frame_index(extend_right_time, self.version)

            # Store original source length for post-processing
            self.src_latents_length = source_latent.shape[-1]

            # Calculate max inference length (240 seconds)
            fps = ACEStepLatentUtils.get_fps(self.version)
            self.max_infer_frame_length = int(240 * fps)
            
            # Initialize trimming variables (matching community logic)
            self.left_trim_length = 0
            self.right_trim_length = 0
            self.to_left_pad_gt_latents = None
            self.to_right_pad_gt_latents = None

            # Create extended latent with trimming logic (matching community exactly)
            gt_latents = source_latent
            frame_length = self.left_frames + gt_latents.shape[-1] + self.right_frames

            # Handle left extension with trimming
            if self.left_frames > 0:
                extend_gt_latents = torch.nn.functional.pad(
                    gt_latents, (self.left_frames, 0), "constant", 0
                )
                if frame_length > self.max_infer_frame_length:
                    self.right_trim_length = frame_length - self.max_infer_frame_length
                    if self.is_v1_5:
                        self.to_right_pad_gt_latents = extend_gt_latents[:, :, -self.right_trim_length:]
                        extend_gt_latents = extend_gt_latents[:, :, :self.max_infer_frame_length]
                    else:
                        self.to_right_pad_gt_latents = extend_gt_latents[:, :, :, -self.right_trim_length:]
                        extend_gt_latents = extend_gt_latents[:, :, :, :self.max_infer_frame_length]
                    frame_length = self.max_infer_frame_length
                gt_latents = extend_gt_latents

            # Handle right extension with trimming
            if self.right_frames > 0:
                extend_gt_latents = torch.nn.functional.pad(
                    gt_latents, (0, self.right_frames), "constant", 0
                )
                frame_length = extend_gt_latents.shape[-1]
                if frame_length > self.max_infer_frame_length:
                    self.left_trim_length = frame_length - self.max_infer_frame_length
                    if self.is_v1_5:
                        self.to_left_pad_gt_latents = extend_gt_latents[:, :, :self.left_trim_length]
                        extend_gt_latents = extend_gt_latents[:, :, -self.max_infer_frame_length:]
                    else:
                        self.to_left_pad_gt_latents = extend_gt_latents[:, :, :, :self.left_trim_length]
                        extend_gt_latents = extend_gt_latents[:, :, :, -self.max_infer_frame_length:]
                    frame_length = self.max_infer_frame_length
                gt_latents = extend_gt_latents

            # Store the working latent (this is what we'll use for inference)
            self.extended_latent = gt_latents

            # Create binary mask for extended regions
            self.extend_mask = torch.zeros_like(self.extended_latent)
            if self.left_frames > 0:
                actual_left_frames = min(self.left_frames, self.extended_latent.shape[-1])
                if self.is_v1_5:
                    self.extend_mask[:, :, :actual_left_frames] = 1.0
                else:
                    self.extend_mask[:, :, :, :actual_left_frames] = 1.0
            if self.right_frames > 0:
                actual_right_frames = min(self.right_frames, self.extended_latent.shape[-1])
                if self.is_v1_5:
                    self.extend_mask[:, :, -actual_right_frames:] = 1.0
                else:
                    self.extend_mask[:, :, :, -actual_right_frames:] = 1.0
        else:
            self.extended_latent = source_latent
            self.extend_mask = None
            self.left_trim_length = 0
            self.right_trim_length = 0
            self.to_left_pad_gt_latents = None
            self.to_right_pad_gt_latents = None
        
        # Setup repaint if enabled
        if self.has_repaint:
            # Adjust repaint times for extended latent
            adjusted_start_time = repaint_start_time + extend_left_time
            adjusted_end_time = repaint_end_time + extend_left_time

            start_frame = ACEStepLatentUtils.time_to_frame_index(adjusted_start_time, self.version)
            end_frame = ACEStepLatentUtils.time_to_frame_index(adjusted_end_time, self.version)

            # Create repaint mask
            self.repaint_mask = torch.zeros_like(self.extended_latent)
            if self.is_v1_5:
                self.repaint_mask[:, :, start_frame:end_frame] = 1.0
            else:
                self.repaint_mask[:, :, :, start_frame:end_frame] = 1.0

            # Apply feathering if requested
            if feather_time > 0:
                feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_time, self.version)
                self.repaint_mask = ACEStepLatentUtils.create_feather_mask(
                    self.repaint_mask, feather_frames)
        else:
            self.repaint_mask = None
        
        # Combine masks: extend OR repaint = generate new content
        self.combined_mask = torch.zeros_like(self.extended_latent)
        if self.extend_mask is not None:
            self.combined_mask = torch.maximum(self.combined_mask, self.extend_mask)
        if self.repaint_mask is not None:
            self.combined_mask = torch.maximum(self.combined_mask, self.repaint_mask)
    
    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to handle hybrid repaint+extend logic"""

        print(f"[HYBRID_GUIDER] sample() called")
        print(f"[HYBRID_GUIDER]   noise.shape: {noise.shape}")
        print(f"[HYBRID_GUIDER]   latent_image.shape: {latent_image.shape}")
        print(f"[HYBRID_GUIDER]   self.is_v1_5: {self.is_v1_5}")
        print(f"[HYBRID_GUIDER]   self.has_extend: {self.has_extend}")
        print(f"[HYBRID_GUIDER]   self.has_repaint: {self.has_repaint}")

        # Use extended latent (which equals source latent if no extend)
        working_latent = self.extended_latent
        print(f"[HYBRID_GUIDER]   working_latent.shape: {working_latent.shape}")

        # Generate noise for working shape
        if self.has_extend:
            if hasattr(noise, 'seed'):
                working_noise = comfy.sample.prepare_noise(working_latent, noise.seed)
            else:
                # Use improved noise generation strategy for better extend blending
                device = noise.device
                dtype = noise.dtype

                # Generate additional noise for extended regions
                retake_noise = torch.randn_like(self.extended_latent, device=device, dtype=dtype)

                # Generate target_latents for the FULL extended duration (matches community)
                target_latents_full = torch.randn_like(self.extended_latent, device=device, dtype=dtype)

                # Create noise by concatenating pieces EXACTLY like community implementation
                # Community does NOT use cosine/sine mixing for extend - only direct concatenation
                padd_list = []

                # Left extension: use retake_noise directly
                if self.left_frames > 0:
                    actual_left_frames = min(self.left_frames, retake_noise.shape[-1])
                    # v1.5 is 3D, v1.0 is 4D - use appropriate indexing
                    if self.is_v1_5:
                        padd_list.append(retake_noise[:, :, :actual_left_frames])
                    else:
                        padd_list.append(retake_noise[:, :, :, :actual_left_frames])

                # Middle section: use TRIMMED SLICE of target_latents_full (key fix!)
                middle_start = self.left_trim_length
                middle_end = target_latents_full.shape[-1] - self.right_trim_length
                # v1.5 is 3D, v1.0 is 4D - use appropriate indexing
                if self.is_v1_5:
                    middle_slice = target_latents_full[:, :, middle_start:middle_end]
                else:
                    middle_slice = target_latents_full[:, :, :, middle_start:middle_end]

                # Trim to source length
                if middle_slice.shape[-1] > self.src_latents_length:
                    if self.is_v1_5:
                        middle_slice = middle_slice[:, :, :self.src_latents_length]
                    else:
                        middle_slice = middle_slice[:, :, :, :self.src_latents_length]
                elif middle_slice.shape[-1] < self.src_latents_length:
                    padding_needed = self.src_latents_length - middle_slice.shape[-1]
                    middle_slice = torch.nn.functional.pad(middle_slice, (0, padding_needed), "constant", 0.0)

                padd_list.append(middle_slice)

                # Right extension: use retake_noise directly
                if self.right_frames > 0:
                    actual_right_frames = min(self.right_frames, retake_noise.shape[-1])
                    # v1.5 is 3D, v1.0 is 4D - use appropriate indexing
                    if self.is_v1_5:
                        padd_list.append(retake_noise[:, :, -actual_right_frames:])
                    else:
                        padd_list.append(retake_noise[:, :, :, -actual_right_frames:])

                working_noise = torch.cat(padd_list, dim=-1)
                print(f"[HYBRID_GUIDER]   working_noise.shape: {working_noise.shape}")

                # Ensure working_noise matches working_latent shape
                assert working_noise.shape[-1] == working_latent.shape[-1], \
                    f"Shape mismatch: {working_noise.shape=} vs {working_latent.shape=}"
        else:
            working_noise = noise
        
        # Calculate n_min if we have repaint
        n_min = 0
        if self.has_repaint:
            total_steps = len(sigmas) - 1
            n_min = int(total_steps * (1 - self.repaint_strength))
        
        # variables for repaint
        x0 = working_latent
        z0 = working_noise  # FIXED noise, not regenerated each step
        zt_edit = x0.clone()
        timesteps = (sigmas * 1000).long()
        original_dtype = working_noise.dtype
        target_latents = working_noise.clone()
        
        def hybrid_callback(step, x0_unused, x, total_steps):
            """Apply hybrid extend+repaint logic"""
            nonlocal target_latents, zt_edit
            
            device = x.device
            i = step
            t = timesteps[step] if step < len(timesteps) else timesteps[-1]
            
            # If we have repaint, use n_min logic
            if self.has_repaint:
                if i < n_min:
                    # Let normal scheduler work for steps before n_min
                    target_latents = x.clone()
                    return
                elif i == n_min:
                    t_i = t.float() / 1000.0
                    zt_src = (1 - t_i) * x0.to(device) + t_i * z0.to(device)
                    target_latents = zt_edit.to(device) + zt_src - x0.to(device)
                    x[:] = target_latents
                    return
                
                # For steps after n_min: use custom ODE + masking
                if i >= n_min and i > 0:
                    t_i = t.float() / 1000.0
                    if i + 1 < len(timesteps):
                        t_im1 = timesteps[i + 1].float() / 1000.0
                    else:
                        t_im1 = torch.zeros_like(t_i).to(t_i.device)
                    
                    # Dtype conversion and ODE
                    target_latents = target_latents.to(torch.float32)
                    prev_sample = x.to(torch.float32)
                    prev_sample = prev_sample.to(original_dtype)
                    target_latents = prev_sample
                    
                    # Apply masking: CRITICAL - use SAME z0 noise, not random!
                    zt_src = (1 - t_im1) * x0.to(device) + t_im1 * z0.to(device)
                    combined_mask_device = self.combined_mask.to(device)
                    target_latents = torch.where(combined_mask_device == 1.0, target_latents, zt_src)
                    
                    x[:] = target_latents
            else:
                # No repaint, just simple extend logic (if any)
                if step > 0:  # Skip initial step
                    device = x.device
                    working_latent_device = working_latent.to(device)
                    combined_mask_device = self.combined_mask.to(device)
                    
                    # Get current timestep and calculate noisy source
                    current_sigma = sigmas[step] if step < len(sigmas) else 0.0
                    sigma_ratio = current_sigma / sigmas[0] if sigmas[0] > 0 else 0.0
                    
                    # Use FIXED noise z0, not random noise!
                    zt_src = (1.0 - sigma_ratio) * working_latent_device + sigma_ratio * z0.to(device)
                    
                    # Apply combined mask: generate new where mask=1.0, preserve original where mask=0.0
                    x[:] = torch.where(combined_mask_device == 1.0, x, zt_src)
            
            # Call original callback if provided
            if callback is not None:
                return callback(step, x0_unused, x, total_steps)
        
        # Run sampling with working latent and hybrid callback
        result = super().sample(working_noise, working_latent, sampler, sigmas, denoise_mask, hybrid_callback, disable_pbar, seed)
        
        # POST-PROCESSING: Apply community's final concatenation logic (if extend is enabled)
        if self.has_extend:
            if self.to_right_pad_gt_latents is not None:
                result = torch.cat([result, self.to_right_pad_gt_latents.to(result.device)], dim=-1)
            if self.to_left_pad_gt_latents is not None:
                # Fix the bug in community code: should be [to_left_pad_gt_latents, result] not [to_right_pad_gt_latents, result]
                result = torch.cat([self.to_left_pad_gt_latents.to(result.device), result], dim=-1)
        
        return result 