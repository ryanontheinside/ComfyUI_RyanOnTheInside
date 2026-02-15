"""
Shared utilities for ACE-Step guiders (both v1.0 and v1.5).

Contains APG (Adaptive Prompt Guidance), reference audio injection,
ODE repaint callback factory, and the common APG application helper.
"""

import torch
import torch.nn.functional as F
import comfy.model_patcher
from . import logger


# =============================================================================
# APG (Adaptive Prompt Guidance) - from ACE-Step reference implementation
# Reference: https://huggingface.co/ACE-Step/acestep-v15-base/blob/main/apg_guidance.py
# =============================================================================

class MomentumBuffer:
    """Momentum buffer for APG guidance smoothing across denoising steps."""

    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def _apg_project(v0: torch.Tensor, v1: torch.Tensor, dims=[-1]):
    """Project v0 onto v1 and return parallel and orthogonal components."""
    dtype = v0.dtype
    device_type = v0.device.type
    if device_type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()

    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device_type), v0_orthogonal.to(dtype).to(device_type)


def _apg_forward(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
    dims=[-1],
):
    """
    Adaptive Prompt Guidance (APG) - replaces standard CFG.

    Instead of standard CFG (uncond + scale * (cond - uncond)), APG:
    1. Computes guidance direction with momentum smoothing
    2. Clamps guidance norm to prevent explosion
    3. Projects guidance perpendicular to conditional prediction
    4. Applies scaled perpendicular guidance

    This prevents the guided result from drifting off the valid latent manifold,
    which is the likely cause of silence with standard CFG at high scales.
    """
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = _apg_project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


def _create_apg_sampler_cfg_function(momentum_buffer):
    """
    Create a sampler_cfg_function that uses APG instead of standard CFG.

    ComfyUI's cfg_function provides args["cond"] = x - denoised = v * sigma (noise pred).
    The sampler_cfg_function must return a noise prediction (same domain).

    For ComfyUI format [B, C, T], dims=[-1] normalizes along T (time),
    matching the reference's dims=[1] for [B, T, C] format.
    """
    def apg_cfg_function(args):
        cond = args["cond"]       # v_cond * sigma (noise prediction domain)
        uncond = args["uncond"]   # v_uncond * sigma
        sigma = args["sigma"]
        guidance_scale = args["cond_scale"]

        # Convert from noise prediction to velocity domain (matching reference)
        sigma_view = sigma.view(sigma.shape[:1] + (1,) * (cond.ndim - 1))
        sigma_clamped = sigma_view.clamp(min=1e-8)
        v_cond = cond / sigma_clamped
        v_uncond = uncond / sigma_clamped

        # APG on velocity (matches reference exactly)
        v_guided = _apg_forward(
            pred_cond=v_cond,
            pred_uncond=v_uncond,
            guidance_scale=guidance_scale,
            momentum_buffer=momentum_buffer,
            eta=0.0,
            norm_threshold=2.5,
            dims=[-1],
        )

        # Convert back to noise prediction domain
        return v_guided * sigma_clamped

    return apg_cfg_function


def _apply_apg_guidance(guider):
    """Apply APG guidance to a guider if cfg > 1.0."""
    if guider.cfg > 1.0:
        momentum_buffer = MomentumBuffer(momentum=-0.75)
        apg_cfg_fn = _create_apg_sampler_cfg_function(momentum_buffer)
        guider.model_options = comfy.model_patcher.create_model_options_clone(guider.model_options)
        guider.model_options["sampler_cfg_function"] = apg_cfg_fn


# =============================================================================
# Reference Audio Timbre Injection Helper
# =============================================================================

def _inject_reference_audio(c, reference_latent, input_batch_size, device, dtype):
    """Inject reference audio timbre latent into model conditioning dict.

    Overrides the silence latent that extra_conds places in c["refer_audio"],
    providing real timbre information to the timbre encoder without triggering
    the covers-mode logic in extra_conds.

    Truncates or pads the reference to match the existing refer_audio length,
    and handles CFG batching.
    """
    ref = reference_latent.to(device=device, dtype=dtype)
    # Match length to whatever extra_conds already set (sized to noise length)
    if "refer_audio" in c:
        target_len = c["refer_audio"].shape[2]
    else:
        target_len = ref.shape[2]
    ref_len = ref.shape[2]
    if ref_len > target_len:
        ref = ref[:, :, :target_len]
    elif ref_len < target_len:
        ref = torch.nn.functional.pad(ref, (0, target_len - ref_len))
    # Handle CFG batching
    if ref.shape[0] < input_batch_size:
        ref = ref.repeat(input_batch_size, 1, 1)
    c["refer_audio"] = ref


# =============================================================================
# ODE Repaint Callback Helper (used by v1.0 guiders)
# =============================================================================

def _make_ode_repaint_callback(x0, z0, repaint_mask, timesteps, n_min, original_dtype, callback=None):
    """Create an ODE repaint callback for v1.0 guiders.

    Implements the 3-phase ODE logic shared by repaint, extend, and hybrid guiders:
    1. Pre-n_min: let normal scheduler work
    2. At n_min: initialize target_latents from source + noise blend
    3. Post-n_min: custom ODE step with mask-based source preservation
    """
    target_latents = [z0.clone()]
    zt_edit = x0.clone()

    def ode_callback(step, x0_unused, x, total_steps):
        device = x.device
        i = step
        t = timesteps[step] if step < len(timesteps) else timesteps[-1]

        if i < n_min:
            target_latents[0] = x.clone()
            return
        elif i == n_min:
            t_i = t.float() / 1000.0
            zt_src = (1 - t_i) * x0.to(device) + t_i * z0.to(device)
            target_latents[0] = zt_edit.to(device) + zt_src - x0.to(device)
            x[:] = target_latents[0]
            if callback is not None:
                return callback(step, x0_unused, x, total_steps)
            return

        if i >= n_min and i > 0:
            t_i = t.float() / 1000.0
            if i + 1 < len(timesteps):
                t_im1 = timesteps[i + 1].float() / 1000.0
            else:
                t_im1 = torch.zeros_like(t_i).to(t_i.device)

            target_latents[0] = target_latents[0].to(torch.float32)
            prev_sample = x.to(torch.float32).to(original_dtype)
            target_latents[0] = prev_sample

            zt_src = (1 - t_im1) * x0.to(device) + t_im1 * z0.to(device)
            repaint_mask_device = repaint_mask.to(device)
            target_latents[0] = torch.where(repaint_mask_device == 1.0, target_latents[0], zt_src)

            x[:] = target_latents[0]

        if callback is not None:
            return callback(step, x0_unused, x, total_steps)

    return ode_callback
