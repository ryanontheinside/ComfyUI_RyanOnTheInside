"""
Monkey patches for ACE-Step 1.5 support in ComfyUI.

These patches extend ComfyUI's ACE-Step implementation to support all task types:
- text2music: Pure generation from text
- repaint: Regenerate a region while preserving the rest
- extend: Add new audio before/after existing audio
- cover: Style transfer/regeneration with target audio as context
- extract: Extract specific track (e.g., vocals, drums) from audio
- lego: Generate specific track within a region

Mask patches (1D latent fixes) are applied at import time.
ACE-Step model patches (forward/tokenizer) are applied lazily on first use.
"""

import torch
import math
import functools
from . import logger

# Flags to track if patches have been applied
_mask_patches_applied = False
_acestep_patches_applied = False

# Task instructions from official ACE-Step 1.5 constants
TASK_INSTRUCTIONS = {
    "text2music": "Fill the audio semantic mask based on the given conditions:",
    "repaint": "Repaint the mask area based on the given conditions:",
    "cover": "Generate audio semantic tokens based on the given conditions:",
    "extract": "Extract the {TRACK_NAME} track from the audio:",
    "extract_default": "Extract the track from the audio:",
    "lego": "Generate the {TRACK_NAME} track based on the audio context:",
    "lego_default": "Generate the track based on the audio context:",
    "complete": "Complete the input track with {TRACK_CLASSES}:",
    "complete_default": "Complete the input track:",
}

# Valid track names for extract/lego tasks
TRACK_NAMES = [
    "woodwinds", "brass", "fx", "synth", "strings", "percussion",
    "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"
]


def get_task_instruction(task_type: str, track_name: str = None) -> str:
    """
    Get the appropriate instruction for a task type.

    Args:
        task_type: One of text2music, repaint, cover, extract, lego, complete
        track_name: Required for extract/lego tasks (e.g., "vocals", "drums")

    Returns:
        The instruction string for the task
    """
    if task_type not in TASK_INSTRUCTIONS and f"{task_type}_default" not in TASK_INSTRUCTIONS:
        raise ValueError(f"Unknown task type: {task_type}. Valid types: {list(TASK_INSTRUCTIONS.keys())}")

    instruction = TASK_INSTRUCTIONS.get(task_type, TASK_INSTRUCTIONS.get(f"{task_type}_default", ""))

    # Handle track name placeholder for extract/lego
    if "{TRACK_NAME}" in instruction:
        if track_name:
            instruction = instruction.format(TRACK_NAME=track_name.upper())
        else:
            # Fall back to default instruction without track name
            instruction = TASK_INSTRUCTIONS.get(f"{task_type}_default", instruction.replace("{TRACK_NAME}", ""))

    return instruction


def _create_patched_ace_step15_forward(original_forward):
    """
    Create a patched forward method for AceStepConditionGenerationModel.

    The patched version:
    1. Accepts chunk_masks, src_latents, is_covers, and precomputed_lm_hints_25Hz as kwargs
    2. Allows guiders to explicitly control is_covers flag for different task types
    3. Properly transposes user-provided tensors from (batch, channels, length) to (batch, length, channels)

    Task type behavior via is_covers:
    - is_covers=0: Use provided src_latents directly (extend/repaint)
    - is_covers=1: Use lm_hints from precomputed_lm_hints_25Hz or audio_codes (cover/extract)
    - is_covers=None: Default behavior (src_latents = x)

    This enables extend, repaint, cover, extract, and lego task types.
    """
    # One-time logging flag
    _forward_logged = [False]

    # Stock prepare_condition uses `is_covers is True` identity checks which fail
    # for tensors. Detect at patch-creation time whether the stock code has been
    # updated to use tensor ops (e.g. unsqueeze) so we pass the right type.
    import inspect
    try:
        _prepare_src = inspect.getsource(
            __import__('comfy.ldm.ace.ace_step15', fromlist=['AceStepConditionGenerationModel'])
            .AceStepConditionGenerationModel.prepare_condition
        )
        _prepare_wants_tensor = "unsqueeze" in _prepare_src
    except Exception:
        _prepare_wants_tensor = False

    @functools.wraps(original_forward)
    def patched_forward(self, x, timestep, context, lyric_embed=None, refer_audio=None,
                        audio_codes=None, chunk_masks=None, src_latents=None,
                        is_covers=None, precomputed_lm_hints_25Hz=None,
                        replace_with_null_embeds=False, **kwargs):
        text_attention_mask = None
        lyric_attention_mask = None
        refer_audio_order_mask = None
        attention_mask = None

        if not hasattr(patched_forward, '_diag_count'):
            patched_forward._diag_count = 0
        patched_forward._diag_count += 1
        _call = patched_forward._diag_count
        # Each step has 2 calls (cond + uncond). Log early, mid, late.
        # Steps 1-2 = calls 1-4, step 15 = calls 29-30, step 28-30 = calls 55-60
        _should_log = False #_call <= 4 or _call in (29, 30) or _call >= 55
        if _should_log:
            print(f"[PATCHED_FWD] Call #{_call}: "
                  f"x={x.shape}, t={timestep}, is_covers={is_covers}, "
                  f"hints={'yes' if precomputed_lm_hints_25Hz is not None else 'no'}, "
                  f"null_embeds={replace_with_null_embeds}")

        # One-time diagnostic log for cover task
        if not _forward_logged[0] and precomputed_lm_hints_25Hz is not None:
            logger.debug(f"[ACE15_PATCHED_FORWARD] Cover mode detected")
            logger.debug(f"[ACE15_PATCHED_FORWARD]   x.shape (before transpose): {x.shape}")
            logger.debug(f"[ACE15_PATCHED_FORWARD]   is_covers: {is_covers}")
            logger.debug(f"[ACE15_PATCHED_FORWARD]   precomputed_lm_hints_25Hz.shape (before transpose): {precomputed_lm_hints_25Hz.shape}")
            logger.debug(f"[ACE15_PATCHED_FORWARD]   precomputed_lm_hints_25Hz stats: mean={precomputed_lm_hints_25Hz.mean():.4f}, std={precomputed_lm_hints_25Hz.std():.4f}")
            _forward_logged[0] = True

        # is_covers can now be passed explicitly by guiders:
        # - Cover/Extract tasks: is_covers=1 (use lm_hints from precomputed_lm_hints_25Hz)
        # - Extend/Repaint tasks: is_covers=0 (use provided src_latents)
        # - Default (not provided): fallback to old behavior
        if is_covers is None and src_latents is not None:
            # Backwards compatibility: default to 0 when src_latents provided but is_covers not specified
            is_covers = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)

        lyric_hidden_states = lyric_embed
        text_hidden_states = context
        refer_audio_acoustic_hidden_states_packed = refer_audio.movedim(-1, -2)

        x = x.movedim(-1, -2)

        if refer_audio_order_mask is None:
            refer_audio_order_mask = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)

        # Transpose user-provided tensors from ComfyUI format (batch, channels, length)
        # to model format (batch, length, channels)
        if src_latents is not None:
            src_latents = src_latents.movedim(-1, -2)
        if chunk_masks is not None:
            chunk_masks = chunk_masks.movedim(-1, -2)
        if precomputed_lm_hints_25Hz is not None:
            # precomputed_lm_hints_25Hz comes in ComfyUI format [B, D, T], transpose to [B, T, D]
            precomputed_lm_hints_25Hz = precomputed_lm_hints_25Hz.movedim(-1, -2)
            # Log post-transpose shape (one-time)
            if _forward_logged[0] and not hasattr(patched_forward, '_post_transpose_logged'):
                logger.debug(f"[ACE15_PATCHED_FORWARD]   precomputed_lm_hints_25Hz.shape (after transpose): {precomputed_lm_hints_25Hz.shape}")
                logger.debug(f"[ACE15_PATCHED_FORWARD]   x.shape (after transpose): {x.shape}")
                logger.debug(f"[ACE15_PATCHED_FORWARD]   src_latents.shape (after transpose): {src_latents.shape if src_latents is not None else 'None'}")
                patched_forward._post_transpose_logged = True

        if src_latents is None:
            src_latents = x

        if chunk_masks is None:
            chunk_masks = torch.ones_like(x)

        # Adapt is_covers type to match what stock prepare_condition expects
        if _prepare_wants_tensor:
            # New stock code expects a tensor — ensure we have one
            if is_covers is True:
                is_covers = torch.ones((x.shape[0],), device=x.device, dtype=torch.long)
            elif is_covers is False or is_covers is None:
                is_covers = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        else:
            # Old stock code uses identity checks — convert truthy tensor to Python bool
            # so the lm_hints branch triggers. Falsy (extend/repaint) stays as tensor
            # so neither branch triggers and src_latents is left unchanged.
            if isinstance(is_covers, torch.Tensor) and is_covers.any().item():
                is_covers = True

        if patched_forward._diag_count <= 2 and precomputed_lm_hints_25Hz is not None:
            print(f"[PATCHED_FWD]   is_covers_adapted={is_covers}, hints.shape={precomputed_lm_hints_25Hz.shape}")

        enc_hidden, enc_mask, context_latents = self.prepare_condition(
            text_hidden_states, text_attention_mask,
            lyric_hidden_states, lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask,
            src_latents, chunk_masks, is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            audio_codes=audio_codes
        )

        # Apply learned null embeddings for uncond CFG pass (matching stock forward behavior)
        if replace_with_null_embeds:
            enc_hidden[:] = self.null_condition_emb.to(enc_hidden)

        if _should_log:
            print(f"[PATCHED_FWD]   enc_hidden: std={enc_hidden.std():.4f}, ctx_latents: shape={context_latents.shape}")
            print(f"[PATCHED_FWD]   x stats: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")

        out = self.decoder(
            hidden_states=x,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
            context_latents=context_latents
        )

        if _should_log:
            print(f"[PATCHED_FWD]   decoder out: mean={out.mean():.4f}, std={out.std():.4f}, min={out.min():.4f}, max={out.max():.4f}")

        return out.movedim(-1, -2)

    return patched_forward


def _create_patched_tokenizer(original_tokenize_with_weights):
    """
    Create a patched tokenize_with_weights method for ACE15Tokenizer.

    The patched version:
    1. Calls the original (stock) tokenizer first to get a baseline output with all
       keys that encode_token_weights expects — this keeps us forward-compatible
       when upstream adds new lm_metadata fields.
    2. Overrides the instruction-dependent keys (lm_prompt, lm_prompt_negative,
       lyrics, qwen3_06b) with task-specific versions.
    3. Overrides generate_audio_codes based on task type (False for cover/extract/lego).
    4. Adds task_type and track_name for downstream use.
    """
    @functools.wraps(original_tokenize_with_weights)
    def patched_tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        task_type = kwargs.get("task_type", "text2music")
        track_name = kwargs.get("track_name", None)

        # Strip our custom kwargs before forwarding to the stock tokenizer.
        # IMPORTANT: copy the dict because the stock tokenizer pop()s keys like
        # bpm, duration, keyscale, timesignature — we still need them afterwards.
        stock_kwargs = {k: v for k, v in kwargs.items() if k not in ("task_type", "track_name")}
        logger.debug(f"[ACE15_TOKENIZER] task={task_type}, track={track_name}")

        # Call the original tokenizer to get baseline output with all expected keys
        out = original_tokenize_with_weights(self, text, return_word_ids, **stock_kwargs.copy())
        logger.debug(f"[ACE15_TOKENIZER] generate_audio_codes={out.get('lm_metadata', {}).get('generate_audio_codes', 'N/A')}")

        # For text2music, the stock tokenizer output is correct — don't override.
        # We only need custom templates for our extended task types.
        if task_type == "text2music":
            out["task_type"] = task_type
            out["track_name"] = track_name
            return out

        # Get appropriate instruction for task type
        instruction = get_task_instruction(task_type, track_name)

        # Re-extract the params we need for our templates
        lyrics = kwargs.get("lyrics", "")
        bpm = kwargs.get("bpm", 120)
        duration = math.ceil(kwargs.get("duration", 120))
        keyscale = kwargs.get("keyscale", "C major")
        timesignature = kwargs.get("timesignature", 2)
        language = kwargs.get("language", "en")

        meta_lm = 'bpm: {}\nduration: {}\nkeyscale: {}\ntimesignature: {}'.format(bpm, duration, keyscale, timesignature)
        lm_template = "<|im_start|>system\n# Instruction\n{}\n\n<|im_end|>\n<|im_start|>user\n# Caption\n{}\n{}\n<|im_end|>\n<|im_start|>assistant\n<think>\n{}\n</think>\n\n<|im_end|>\n"
        meta_cap = '- bpm: {}\n- timesignature: {}\n- keyscale: {}\n- duration: {}\n'.format(bpm, timesignature, keyscale, duration)

        # Override only the instruction-dependent tokenized outputs
        out["lm_prompt"] = self.qwen3_06b.tokenize_with_weights(lm_template.format(instruction, text, lyrics, meta_lm), disable_weights=True)
        out["lm_prompt_negative"] = self.qwen3_06b.tokenize_with_weights(lm_template.format(instruction, text, lyrics, ""), disable_weights=True)
        out["lyrics"] = self.qwen3_06b.tokenize_with_weights("# Languages\n{}\n\n# Lyric{}<|endoftext|><|endoftext|>".format(language, lyrics), return_word_ids, disable_weights=True, **stock_kwargs.copy())
        _fmt = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n".format(instruction, text, meta_cap)
        logger.debug(f"[ACE15_TOKENIZER] qwen3_06b prompt (first 200 chars): {repr(_fmt[:200])}")
        out["qwen3_06b"] = self.qwen3_06b.tokenize_with_weights(_fmt, return_word_ids, **stock_kwargs.copy())

        # Cover/extract/lego tasks use precomputed semantic hints, skip LLM code generation
        if isinstance(out.get("lm_metadata"), dict):
            out["lm_metadata"]["generate_audio_codes"] = task_type in ("text2music", "repaint")

        # Store task info for downstream use
        out["task_type"] = task_type
        out["track_name"] = track_name

        logger.debug(f"[ACE15_TOKENIZER] final generate_audio_codes={out.get('lm_metadata', {}).get('generate_audio_codes', 'N/A')}")
        return out

    return patched_tokenize_with_weights


def _apply_mask_patches():
    """
    Apply mask-related patches for 1D latent support.

    These are safe to apply at import time since they only fix ComfyUI bugs
    for 1D spatial dimensions and don't change behavior for standard 2D/3D latents.
    """
    global _mask_patches_applied

    if _mask_patches_applied:
        return

    try:
        # Patch resolve_areas_and_cond_masks_multidim to handle 1D latents (ACEStep15)
        # ComfyUI's stock implementation assumes at least 2 spatial dims (dims[-2]),
        # which fails for 1D audio latents where dims is a 1-element tuple.
        import comfy.samplers as _samplers
        _original_resolve = _samplers.resolve_areas_and_cond_masks_multidim

        def _patched_resolve_areas_and_cond_masks_multidim(conditions, dims, device):
            for i in range(len(conditions)):
                c = conditions[i]
                if 'area' in c:
                    area = c['area']
                    if area[0] == "percentage":
                        modified = c.copy()
                        a = area[1:]
                        a_len = len(a) // 2
                        area = ()
                        for d in range(len(dims)):
                            area += (max(1, round(a[d] * dims[d])),)
                        for d in range(len(dims)):
                            area += (round(a[d + a_len] * dims[d]),)
                        modified['area'] = area
                        c = modified
                        conditions[i] = c

                if 'mask' in c:
                    mask = c['mask']
                    mask = mask.to(device=device)
                    modified = c.copy()
                    # Normalize mask to [batch, *spatial_dims]
                    target_ndim = len(dims) + 1
                    while mask.ndim > target_ndim and mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                    while mask.ndim < target_ndim:
                        mask = mask.unsqueeze(0)
                    if mask.shape[1:] != dims:
                        if len(dims) == 1:
                            mask = torch.nn.functional.interpolate(
                                mask.unsqueeze(1), size=dims[0],
                                mode='linear', align_corners=False).squeeze(1)
                        elif mask.ndim < 4:
                            import comfy.utils
                            mask = comfy.utils.common_upscale(mask.unsqueeze(1), dims[-1], dims[-2], 'bilinear', 'none').squeeze(1)
                        else:
                            import comfy.utils
                            mask = comfy.utils.common_upscale(mask, dims[-1], dims[-2], 'bilinear', 'none')

                    if modified.get("set_area_to_bounds", False) and len(dims) == 2:
                        from comfy.samplers import get_mask_aabb
                        bounds = torch.max(torch.abs(mask), dim=0).values.unsqueeze(0)
                        boxes, is_empty = get_mask_aabb(bounds)
                        if is_empty[0]:
                            modified['area'] = (8, 8, 0, 0)
                        else:
                            box = boxes[0]
                            H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
                            H = max(8, H)
                            W = max(8, W)
                            modified['area'] = (int(H), int(W), int(Y), int(X))

                    modified['mask'] = mask
                    conditions[i] = modified

        _samplers.resolve_areas_and_cond_masks_multidim = _patched_resolve_areas_and_cond_masks_multidim
        logger.info("[ACE-Step Patches] Patched resolve_areas_and_cond_masks_multidim for 1D latent support")
    except Exception as e:
        logger.warning(f"[ACE-Step Patches] Warning: Could not patch resolve_areas_and_cond_masks_multidim: {e}")

    try:
        # Patch comfy.utils.reshape_mask to handle 1D latents (ACEStep15)
        # Stock reshape_mask sets scale_mode='linear' for 1D but doesn't reshape
        # the input to (batch, 1, length) like it does for 2D/3D cases.
        import comfy.utils as _utils

        _original_reshape_mask = _utils.reshape_mask

        def _patched_reshape_mask(input_mask, output_shape):
            dims = len(output_shape) - 2
            if dims == 1:
                input_mask = input_mask.reshape((-1, 1, input_mask.shape[-1]))
                # Fall through to stock interpolation logic would be ideal,
                # but we can't since we replaced the function. Replicate it:
                mask = torch.nn.functional.interpolate(input_mask, size=output_shape[2:], mode='linear')
                if mask.shape[1] < output_shape[1]:
                    mask = mask.repeat((1, output_shape[1]) + (1,) * dims)[:, :output_shape[1]]
                mask = _utils.repeat_to_batch_size(mask, output_shape[0])
                return mask
            return _original_reshape_mask(input_mask, output_shape)

        _utils.reshape_mask = _patched_reshape_mask
        # Also patch the reference in sampler_helpers since it imports comfy.utils at module level
        import comfy.sampler_helpers as _sampler_helpers
        _sampler_helpers.comfy.utils.reshape_mask = _patched_reshape_mask
        logger.info("[ACE-Step Patches] Patched comfy.utils.reshape_mask for 1D latent support")
    except Exception as e:
        logger.warning(f"[ACE-Step Patches] Warning: Could not patch reshape_mask: {e}")

    _mask_patches_applied = True


def apply_acestep_patches():
    """
    Apply ACE-Step model patches (forward method and tokenizer).

    These are applied lazily — only when an ACE-Step node is actually executed —
    so that users who have this node pack installed but aren't using ACE-Step
    nodes don't get their ACE-Step forward method overwritten.

    Safe to call multiple times (will only apply once).
    """
    global _acestep_patches_applied

    if _acestep_patches_applied:
        return

    try:
        # Patch AceStepConditionGenerationModel.forward
        from comfy.ldm.ace.ace_step15 import AceStepConditionGenerationModel
        original_forward = AceStepConditionGenerationModel.forward
        AceStepConditionGenerationModel.forward = _create_patched_ace_step15_forward(original_forward)
        logger.info("[ACE-Step Patches] Patched AceStepConditionGenerationModel.forward")
    except ImportError as e:
        logger.info(f"[ACE-Step Patches] Warning: Could not patch ace_step15: {e}")

    try:
        # Patch ACE15Tokenizer.tokenize_with_weights
        from comfy.text_encoders.ace15 import ACE15Tokenizer
        original_tokenize = ACE15Tokenizer.tokenize_with_weights
        ACE15Tokenizer.tokenize_with_weights = _create_patched_tokenizer(original_tokenize)
        logger.info("[ACE-Step Patches] Patched ACE15Tokenizer.tokenize_with_weights")
    except ImportError as e:
        logger.info(f"[ACE-Step Patches] Warning: Could not patch ace15 tokenizer: {e}")

    _acestep_patches_applied = True
    logger.info("[ACE-Step Patches] ACE-Step model patches applied")


def is_patched():
    """Check if ACE-Step model patches have been applied."""
    return _acestep_patches_applied


# Apply mask patches at import time (safe for all users, fixes 1D latent bugs)
_apply_mask_patches()
