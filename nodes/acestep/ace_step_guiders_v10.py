"""
ACE-Step 1.0 guiders for ComfyUI.

These guiders use latent-level ODE blending for repaint, extend, and hybrid operations.
They work with v1.0 latents (batch, 8, 16, length) and v1.5 latents (batch, 64, length).

NOTE: gigantic shoutout to the powerful acestep team https://github.com/ace-step/ACE-Step
NOTE: And another massive shoutout.... the adapted code here is based on https://github.com/billwuhao/ComfyUI_ACE-Step

NOTE: this implementation is experimental and beta with a minimum of testing
"""

import torch
import comfy.samplers
import comfy.sample
import comfy.model_patcher
from .ace_step_utils import ACEStepLatentUtils
from . import logger
from .ace_step_guiders_common import _apply_apg_guidance, _make_ode_repaint_callback


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

        self.retake_variance = repaint_strength

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to implement repaint logic"""

        logger.debug(f"[REPAINT_GUIDER] sample() called")
        logger.debug(f"[REPAINT_GUIDER]   noise.shape: {noise.shape}")
        logger.debug(f"[REPAINT_GUIDER]   self.source_latent.shape: {self.source_latent.shape}")

        total_steps = len(sigmas) - 1
        n_min = int(total_steps * (1 - self.retake_variance))
        timesteps = (sigmas * 1000).long()

        repaint_callback = _make_ode_repaint_callback(
            x0=self.source_latent,
            z0=noise,
            repaint_mask=self.repaint_mask,
            timesteps=timesteps,
            n_min=n_min,
            original_dtype=noise.dtype,
            callback=callback,
        )

        _apply_apg_guidance(self)

        return super().sample(noise, latent_image, sampler, sigmas, denoise_mask, repaint_callback, disable_pbar, seed)


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

        self.retake_variance = 1.0

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to implement extend logic exactly matching community implementation"""

        logger.debug(f"[EXTEND_GUIDER] sample() called")
        logger.debug(f"[EXTEND_GUIDER]   noise.shape: {noise.shape}")
        logger.debug(f"[EXTEND_GUIDER]   self.extended_latent.shape: {self.extended_latent.shape}")

        device = noise.device
        dtype = noise.dtype

        retake_latents = torch.randn_like(self.extended_latent, device=device, dtype=dtype)
        target_latents_full = torch.randn_like(self.extended_latent, device=device, dtype=dtype)

        padd_list = []

        if self.left_frames > 0:
            actual_left_frames = min(self.left_frames, retake_latents.shape[-1])
            if self.is_v1_5:
                padd_list.append(retake_latents[:, :, :actual_left_frames])
            else:
                padd_list.append(retake_latents[:, :, :, :actual_left_frames])

        middle_start = self.left_trim_length
        middle_end = target_latents_full.shape[-1] - self.right_trim_length
        if self.is_v1_5:
            middle_slice = target_latents_full[:, :, middle_start:middle_end]
        else:
            middle_slice = target_latents_full[:, :, :, middle_start:middle_end]

        if middle_slice.shape[-1] > self.src_latents_length:
            if self.is_v1_5:
                middle_slice = middle_slice[:, :, :self.src_latents_length]
            else:
                middle_slice = middle_slice[:, :, :, :self.src_latents_length]
        elif middle_slice.shape[-1] < self.src_latents_length:
            padding_needed = self.src_latents_length - middle_slice.shape[-1]
            middle_slice = torch.nn.functional.pad(middle_slice, (0, padding_needed), "constant", 0.0)

        padd_list.append(middle_slice)

        if self.right_frames > 0:
            actual_right_frames = min(self.right_frames, retake_latents.shape[-1])
            if self.is_v1_5:
                padd_list.append(retake_latents[:, :, -actual_right_frames:])
            else:
                padd_list.append(retake_latents[:, :, :, -actual_right_frames:])

        target_latents = torch.cat(padd_list, dim=-1)

        if target_latents.shape[-1] != self.extended_latent.shape[-1]:
            raise ValueError(
                f"Shape mismatch: target_latents {target_latents.shape} vs extended_latent {self.extended_latent.shape}"
            )

        total_steps = len(sigmas) - 1
        n_min = int(total_steps * (1 - self.retake_variance))
        timesteps = (sigmas * 1000).long()

        extend_callback = _make_ode_repaint_callback(
            x0=self.extended_latent,
            z0=target_latents,
            repaint_mask=self.extend_mask,
            timesteps=timesteps,
            n_min=n_min,
            original_dtype=target_latents.dtype,
            callback=callback,
        )

        _apply_apg_guidance(self)

        result = super().sample(target_latents, self.extended_latent, sampler, sigmas, denoise_mask, extend_callback, disable_pbar, seed)

        # POST-PROCESSING: Apply community's final concatenation logic
        if self.to_right_pad_gt_latents is not None:
            result = torch.cat([result, self.to_right_pad_gt_latents.to(result.device)], dim=-1)
        if self.to_left_pad_gt_latents is not None:
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

        # Setup extend if enabled
        if self.has_extend:
            self.left_frames = ACEStepLatentUtils.time_to_frame_index(extend_left_time, self.version)
            self.right_frames = ACEStepLatentUtils.time_to_frame_index(extend_right_time, self.version)
            self.src_latents_length = source_latent.shape[-1]

            fps = ACEStepLatentUtils.get_fps(self.version)
            self.max_infer_frame_length = int(240 * fps)

            self.left_trim_length = 0
            self.right_trim_length = 0
            self.to_left_pad_gt_latents = None
            self.to_right_pad_gt_latents = None

            gt_latents = source_latent
            frame_length = self.left_frames + gt_latents.shape[-1] + self.right_frames

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

            self.extended_latent = gt_latents

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
            adjusted_start_time = repaint_start_time + extend_left_time
            adjusted_end_time = repaint_end_time + extend_left_time

            start_frame = ACEStepLatentUtils.time_to_frame_index(adjusted_start_time, self.version)
            end_frame = ACEStepLatentUtils.time_to_frame_index(adjusted_end_time, self.version)

            self.repaint_mask = torch.zeros_like(self.extended_latent)
            if self.is_v1_5:
                self.repaint_mask[:, :, start_frame:end_frame] = 1.0
            else:
                self.repaint_mask[:, :, :, start_frame:end_frame] = 1.0

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

        logger.debug(f"[HYBRID_GUIDER] sample() called")
        logger.debug(f"[HYBRID_GUIDER]   has_extend: {self.has_extend}, has_repaint: {self.has_repaint}")

        working_latent = self.extended_latent

        # Generate noise for working shape
        if self.has_extend:
            if hasattr(noise, 'seed'):
                working_noise = comfy.sample.prepare_noise(working_latent, noise.seed)
            else:
                device = noise.device
                dtype = noise.dtype

                retake_noise = torch.randn_like(self.extended_latent, device=device, dtype=dtype)
                target_latents_full = torch.randn_like(self.extended_latent, device=device, dtype=dtype)

                padd_list = []

                if self.left_frames > 0:
                    actual_left_frames = min(self.left_frames, retake_noise.shape[-1])
                    if self.is_v1_5:
                        padd_list.append(retake_noise[:, :, :actual_left_frames])
                    else:
                        padd_list.append(retake_noise[:, :, :, :actual_left_frames])

                middle_start = self.left_trim_length
                middle_end = target_latents_full.shape[-1] - self.right_trim_length
                if self.is_v1_5:
                    middle_slice = target_latents_full[:, :, middle_start:middle_end]
                else:
                    middle_slice = target_latents_full[:, :, :, middle_start:middle_end]

                if middle_slice.shape[-1] > self.src_latents_length:
                    if self.is_v1_5:
                        middle_slice = middle_slice[:, :, :self.src_latents_length]
                    else:
                        middle_slice = middle_slice[:, :, :, :self.src_latents_length]
                elif middle_slice.shape[-1] < self.src_latents_length:
                    padding_needed = self.src_latents_length - middle_slice.shape[-1]
                    middle_slice = torch.nn.functional.pad(middle_slice, (0, padding_needed), "constant", 0.0)

                padd_list.append(middle_slice)

                if self.right_frames > 0:
                    actual_right_frames = min(self.right_frames, retake_noise.shape[-1])
                    if self.is_v1_5:
                        padd_list.append(retake_noise[:, :, -actual_right_frames:])
                    else:
                        padd_list.append(retake_noise[:, :, :, -actual_right_frames:])

                working_noise = torch.cat(padd_list, dim=-1)

                if working_noise.shape[-1] != working_latent.shape[-1]:
                    raise ValueError(
                        f"Shape mismatch: working_noise {working_noise.shape} vs working_latent {working_latent.shape}"
                    )
        else:
            working_noise = noise

        # Calculate n_min if we have repaint
        n_min = 0
        if self.has_repaint:
            total_steps = len(sigmas) - 1
            n_min = int(total_steps * (1 - self.repaint_strength))

        timesteps = (sigmas * 1000).long()

        # Hybrid callback handles both extend-only and repaint cases
        x0 = working_latent
        z0 = working_noise
        zt_edit = x0.clone()
        original_dtype = working_noise.dtype
        target_latents = working_noise.clone()

        def hybrid_callback(step, x0_unused, x, total_steps):
            nonlocal target_latents, zt_edit

            device = x.device
            i = step
            t = timesteps[step] if step < len(timesteps) else timesteps[-1]

            if self.has_repaint:
                if i < n_min:
                    target_latents = x.clone()
                    return
                elif i == n_min:
                    t_i = t.float() / 1000.0
                    zt_src = (1 - t_i) * x0.to(device) + t_i * z0.to(device)
                    target_latents = zt_edit.to(device) + zt_src - x0.to(device)
                    x[:] = target_latents
                    return

                if i >= n_min and i > 0:
                    t_i = t.float() / 1000.0
                    if i + 1 < len(timesteps):
                        t_im1 = timesteps[i + 1].float() / 1000.0
                    else:
                        t_im1 = torch.zeros_like(t_i).to(t_i.device)

                    target_latents = target_latents.to(torch.float32)
                    prev_sample = x.to(torch.float32)
                    prev_sample = prev_sample.to(original_dtype)
                    target_latents = prev_sample

                    zt_src = (1 - t_im1) * x0.to(device) + t_im1 * z0.to(device)
                    combined_mask_device = self.combined_mask.to(device)
                    target_latents = torch.where(combined_mask_device == 1.0, target_latents, zt_src)

                    x[:] = target_latents
            else:
                if step > 0:
                    device = x.device
                    working_latent_device = working_latent.to(device)
                    combined_mask_device = self.combined_mask.to(device)

                    current_sigma = sigmas[step] if step < len(sigmas) else 0.0
                    sigma_ratio = current_sigma / sigmas[0] if sigmas[0] > 0 else 0.0

                    zt_src = (1.0 - sigma_ratio) * working_latent_device + sigma_ratio * z0.to(device)
                    x[:] = torch.where(combined_mask_device == 1.0, x, zt_src)

            if callback is not None:
                return callback(step, x0_unused, x, total_steps)

        _apply_apg_guidance(self)

        result = super().sample(working_noise, working_latent, sampler, sigmas, denoise_mask, hybrid_callback, disable_pbar, seed)

        # POST-PROCESSING: Apply community's final concatenation logic
        if self.has_extend:
            if self.to_right_pad_gt_latents is not None:
                result = torch.cat([result, self.to_right_pad_gt_latents.to(result.device)], dim=-1)
            if self.to_left_pad_gt_latents is not None:
                result = torch.cat([self.to_left_pad_gt_latents.to(result.device), result], dim=-1)

        return result
