import torch
import comfy.samplers
import comfy.sample
import math
from .ace_step_utils import ACEStepLatentUtils

# NOTE: gigantic shoutout to the powerful acestep team https://github.com/ace-step/ACE-Step
# NOTE: And another massive shoutout.... the adapted code here is based on https://github.com/billwuhao/ComfyUI_ACE-Step

# NOTE: this implementation is experimental and beta with a minimum of testing

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
        
        # Convert times to frame indices
        self.start_frame = ACEStepLatentUtils.time_to_frame_index(start_time)
        self.end_frame = ACEStepLatentUtils.time_to_frame_index(end_time)
        
        # Create binary repaint mask (1.0 = generate new, 0.0 = preserve original)
        self.repaint_mask = torch.zeros_like(source_latent)
        self.repaint_mask[:, :, :, self.start_frame:self.end_frame] = 1.0
        
        # Apply feathering if requested
        if feather_time > 0:
            feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_time)
            self.repaint_mask = ACEStepLatentUtils.create_feather_mask(
                self.repaint_mask, feather_frames)
        
        # Calculate n_min based on repaint_strength
        self.retake_variance = repaint_strength
    
    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to implement repaint logic"""
        
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
        
        # Calculate frame extensions
        self.left_frames = ACEStepLatentUtils.time_to_frame_index(extend_left_time)
        self.right_frames = ACEStepLatentUtils.time_to_frame_index(extend_right_time)
        
        # Create extended latent by padding with zeros
        self.extended_latent = torch.nn.functional.pad(
            source_latent, 
            (self.left_frames, self.right_frames), 
            "constant", 0.0
        )
        
        # Create binary mask for extended regions
        self.extend_mask = torch.zeros_like(self.extended_latent)
        if self.left_frames > 0:
            self.extend_mask[:, :, :, :self.left_frames] = 1.0
        if self.right_frames > 0:
            self.extend_mask[:, :, :, -self.right_frames:] = 1.0
        
        # Calculate n_min based on retake_variance (using 1.0 for extend)
        self.retake_variance = 1.0
        
    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        """Override sample to implement extend logic with custom ODE"""
        
        # Generate retake_latents (separate noise for extended regions)
        retake_latents = torch.randn_like(self.extended_latent)
        
        # Create initial target_latents by concatenating noise pieces
        padd_list = []
        if self.left_frames > 0:
            left_piece = retake_latents[:, :, :, :self.left_frames]
            padd_list.append(left_piece)
        
        # Middle section: use original noise (trimmed if needed)
        original_noise_start = 0
        original_noise_end = noise.shape[-1]
        middle_piece = noise[:, :, :, original_noise_start:original_noise_end]
        padd_list.append(middle_piece)
        
        if self.right_frames > 0:
            right_piece = retake_latents[:, :, :, -self.right_frames:]
            padd_list.append(right_piece)
        
        target_latents = torch.cat(padd_list, dim=-1)
        
        # Calculate n_min
        total_steps = len(sigmas) - 1
        n_min = int(total_steps * (1 - self.retake_variance))
        
        # Core variables
        x0 = self.extended_latent  # ground truth latents
        z0 = target_latents        # initial noise
        zt_edit = x0.clone()       # will be updated
        is_repaint = True          # extend is treated as repaint
        repaint_mask = self.extend_mask
        
        # Convert sigmas to timesteps
        timesteps = (sigmas * 1000).long()
        
        # Store original dtype for precision
        original_dtype = target_latents.dtype
        
        def extend_callback(step, x0_unused, x, total_steps):
            """Implement extend logic with custom ODE"""
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
                
                # preserve source regions, keep generated in extended regions
                zt_src = (1 - t_im1) * x0.to(device) + t_im1 * z0.to(device)
                extend_mask = repaint_mask.to(device)
                target_latents = torch.where(extend_mask == 1.0, target_latents, zt_src)
                
                # Replace x with our custom ODE result
                x[:] = target_latents
            
            # Call original callback if provided
            if callback is not None:
                return callback(step, x0_unused, x, total_steps)
        
        # Use extended latent and target_latents as initial noise
        result = super().sample(target_latents, self.extended_latent, sampler, sigmas, denoise_mask, extend_callback, disable_pbar, seed)
        
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
        
        # Check which operations are enabled
        self.has_repaint = repaint_start_time is not None and repaint_end_time is not None
        self.has_extend = extend_left_time > 0 or extend_right_time > 0
        
        # Setup extend if enabled
        if self.has_extend:
            self.left_frames = ACEStepLatentUtils.time_to_frame_index(extend_left_time)
            self.right_frames = ACEStepLatentUtils.time_to_frame_index(extend_right_time)
            
            # Create extended latent by padding with zeros
            self.extended_latent = torch.nn.functional.pad(
                source_latent, 
                (self.left_frames, self.right_frames), 
                "constant", 0.0
            )
            
            # Create binary mask for extended regions
            self.extend_mask = torch.zeros_like(self.extended_latent)
            if self.left_frames > 0:
                self.extend_mask[:, :, :, :self.left_frames] = 1.0
            if self.right_frames > 0:
                self.extend_mask[:, :, :, -self.right_frames:] = 1.0
        else:
            self.extended_latent = source_latent
            self.extend_mask = None
        
        # Setup repaint if enabled
        if self.has_repaint:
            # Adjust repaint times for extended latent
            adjusted_start_time = repaint_start_time + extend_left_time
            adjusted_end_time = repaint_end_time + extend_left_time
            
            start_frame = ACEStepLatentUtils.time_to_frame_index(adjusted_start_time)
            end_frame = ACEStepLatentUtils.time_to_frame_index(adjusted_end_time)
            
            # Create repaint mask
            self.repaint_mask = torch.zeros_like(self.extended_latent)
            self.repaint_mask[:, :, :, start_frame:end_frame] = 1.0
            
            # Apply feathering if requested
            if feather_time > 0:
                feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_time)
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
        
        # Use extended latent (which equals source latent if no extend)
        working_latent = self.extended_latent
        
        # Generate noise for working shape
        if self.has_extend:
            if hasattr(noise, 'seed'):
                working_noise = comfy.sample.prepare_noise(working_latent, noise.seed)
            else:
                working_noise = torch.nn.functional.pad(
                    noise, 
                    (self.left_frames, self.right_frames), 
                    "constant", 0.0
                )
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
        return super().sample(working_noise, working_latent, sampler, sigmas, denoise_mask, hybrid_callback, disable_pbar, seed) 