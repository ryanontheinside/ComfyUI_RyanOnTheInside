"""
Monkey patches for ACE-Step 1.5 support in ComfyUI.

These patches extend ComfyUI's ACE-Step implementation to support all task types:
- text2music: Pure generation from text
- repaint: Regenerate a region while preserving the rest
- extend: Add new audio before/after existing audio
- cover: Style transfer/regeneration with target audio as context
- extract: Extract specific track (e.g., vocals, drums) from audio
- lego: Generate specific track within a region

Patches are applied at module import time.
"""

import torch
import math
import functools
from . import logger

# Flag to track if patches have been applied
_patches_applied = False

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

    @functools.wraps(original_forward)
    def patched_forward(self, x, timestep, context, lyric_embed=None, refer_audio=None,
                        audio_codes=None, chunk_masks=None, src_latents=None,
                        is_covers=None, precomputed_lm_hints_25Hz=None, **kwargs):
        text_attention_mask = None
        lyric_attention_mask = None
        refer_audio_order_mask = None
        attention_mask = None

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

        if src_latents is None and is_covers is None:
            src_latents = x

        if chunk_masks is None:
            chunk_masks = torch.ones_like(x)

        # Stock prepare_condition uses identity checks (is_covers is True / is False)
        # which fail for tensors. Convert to Python bool so the lm_hints branch triggers.
        # Only convert truthy tensors — falsy (extend/repaint) must stay as tensor so
        # neither branch triggers and src_latents is left unchanged.
        if isinstance(is_covers, torch.Tensor) and is_covers.any().item():
            is_covers = True

        enc_hidden, enc_mask, context_latents = self.prepare_condition(
            text_hidden_states, text_attention_mask,
            lyric_hidden_states, lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed, refer_audio_order_mask,
            src_latents, chunk_masks, is_covers,
            precomputed_lm_hints_25Hz=precomputed_lm_hints_25Hz,
            audio_codes=audio_codes
        )

        out = self.decoder(
            hidden_states=x,
            timestep=timestep,
            timestep_r=timestep,
            attention_mask=attention_mask,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
            context_latents=context_latents
        )

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

        # Strip our custom kwargs before forwarding to the stock tokenizer
        stock_kwargs = {k: v for k, v in kwargs.items() if k not in ("task_type", "track_name")}

        # Call the original tokenizer to get baseline output with all expected keys
        out = original_tokenize_with_weights(self, text, return_word_ids, **stock_kwargs)

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
        out["lyrics"] = self.qwen3_06b.tokenize_with_weights("# Languages\n{}\n\n# Lyric{}<|endoftext|><|endoftext|>".format(language, lyrics), return_word_ids, disable_weights=True, **kwargs)
        out["qwen3_06b"] = self.qwen3_06b.tokenize_with_weights("# Instruction\n{}\n\n# Caption\n{}# Metas\n{}<|endoftext|>\n<|endoftext|>".format(instruction, text, meta_cap), return_word_ids, **kwargs)

        # Cover/extract/lego tasks use precomputed semantic hints, skip LLM code generation
        if isinstance(out.get("lm_metadata"), dict):
            out["lm_metadata"]["generate_audio_codes"] = task_type in ("text2music", "repaint")

        # Store task info for downstream use
        out["task_type"] = task_type
        out["track_name"] = track_name

        return out

    return patched_tokenize_with_weights


def apply_patches():
    """
    Apply all ACE-Step 1.5 patches.

    This should be called once at module load time.
    Safe to call multiple times (will only apply once).
    """
    global _patches_applied

    if _patches_applied:
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

    _patches_applied = True
    logger.info("[ACE-Step Patches] All patches applied successfully")


def is_patched():
    """Check if patches have been applied."""
    return _patches_applied


# Apply patches when this module is imported
apply_patches()
