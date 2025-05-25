import torch
import torch.nn.functional as F
import math

class ACEStepLatentUtils:
    """Utility functions for ACEStep audio latent manipulation"""
    
    @staticmethod
    def time_to_frame_index(time_seconds):
        """Convert time in seconds to ACE latent frame index
        ACE audio: 1 second = 44100 samples / 512 / 8 ≈ 10.8 frames
        """
        return int(time_seconds * 44100 / 512 / 8)
    
    @staticmethod
    def frame_index_to_time(frame_index):
        """Convert ACE latent frame index to time in seconds"""
        return frame_index * 512 * 8 / 44100
    
    @staticmethod
    def validate_ace_latent_shape(latent):
        """Validate that latent has correct ACE audio shape"""
        if len(latent.shape) != 4:
            raise ValueError(f"ACE latent must be 4D, got {len(latent.shape)}D")
        
        batch_size, channels, height, length = latent.shape
        if channels != 8:
            raise ValueError(f"ACE latent must have 8 channels, got {channels}")
        if height != 16:
            raise ValueError(f"ACE latent must have height 16, got {height}")
        
        return True
    
    @staticmethod
    def create_repaint_mask(latent_shape, start_frame, end_frame):
        """Create a mask for repainting a specific region
        
        Args:
            latent_shape: (batch, 8, 16, length)
            start_frame: Frame index to start repainting
            end_frame: Frame index to end repainting
            
        Returns:
            torch.Tensor: Binary mask with 1.0 in repaint region
        """
        batch_size, channels, height, length = latent_shape
        
        # Clamp frame indices to valid range
        start_frame = max(0, start_frame)
        end_frame = min(length, end_frame)
        
        if start_frame >= end_frame:
            raise ValueError(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")
        
        mask = torch.zeros(latent_shape)
        mask[:, :, :, start_frame:end_frame] = 1.0
        
        return mask
    
    @staticmethod
    def create_extend_mask(source_shape, left_frames, right_frames):
        """Create a mask for extension regions
        
        Args:
            source_shape: (batch, 8, 16, length) shape of source latent
            left_frames: Number of frames to extend on the left
            right_frames: Number of frames to extend on the right
            
        Returns:
            torch.Tensor: Binary mask with 1.0 in extension regions
        """
        batch_size, channels, height, source_length = source_shape
        total_length = source_length + left_frames + right_frames
        
        mask = torch.zeros((batch_size, channels, height, total_length))
        
        # Mark left extension region
        if left_frames > 0:
            mask[:, :, :, :left_frames] = 1.0
        
        # Mark right extension region  
        if right_frames > 0:
            mask[:, :, :, -right_frames:] = 1.0
            
        return mask
    
    @staticmethod
    def create_extend_region_mask(extended_shape, left_frames, right_frames, source_length):
        """Create a mask indicating which regions to generate vs preserve in extended latent
        
        Args:
            extended_shape: Shape of the extended latent (batch, 8, 16, total_length)
            left_frames: Number of frames added on the left
            right_frames: Number of frames added on the right
            source_length: Length of the original source latent
            
        Returns:
            torch.Tensor: Binary mask with 1.0 = generate (extension regions), 0.0 = preserve (source region)
        """
        batch_size, channels, height, total_length = extended_shape
        
        mask = torch.zeros(extended_shape)
        
        # Mark left extension region for generation
        if left_frames > 0:
            mask[:, :, :, :left_frames] = 1.0
        
        # Mark right extension region for generation
        if right_frames > 0:
            mask[:, :, :, -right_frames:] = 1.0
        
        # The middle region (source_length) remains 0.0 for preservation
        
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
            mask: Binary mask tensor
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
        
        # Reshape for 1D convolution along time dimension
        kernel = gaussian_kernel.view(1, 1, 1, -1)
        
        # Apply convolution to each channel separately
        batch_size, channels, height, length = mask.shape
        result = torch.zeros_like(mask)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(height):
                    # Extract 1D signal
                    signal = mask[b:b+1, c:c+1, h:h+1, :]  # Shape: (1, 1, 1, length)
                    
                    # Pad with constant values (edge values)
                    padded_signal = F.pad(signal, (feather_frames, feather_frames), 'constant', 0.0)
                    
                    # Apply 1D convolution
                    smoothed = F.conv2d(padded_signal, kernel, padding=0)
                    
                    # Store result
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
            source_latents: Original audio latents (batch, 8, 16, length)
            start_time: Start time in seconds for repaint region
            end_time: End time in seconds for repaint region
            repaint_strength: Strength of repainting (0.0 = no repaint, 1.0 = full repaint)
            feather_time: Time in seconds for feathering edges
        """
        ACEStepLatentUtils.validate_ace_latent_shape(source_latents)
        
        self.source_latents = source_latents
        self.start_frame = ACEStepLatentUtils.time_to_frame_index(start_time)
        self.end_frame = ACEStepLatentUtils.time_to_frame_index(end_time)
        self.repaint_strength = repaint_strength
        self.feather_frames = ACEStepLatentUtils.time_to_frame_index(feather_time)
        
        # Create masks
        self.repaint_mask = ACEStepLatentUtils.create_repaint_mask(
            source_latents.shape, self.start_frame, self.end_frame)
        
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
            source_latents: Original audio latents (batch, 8, 16, length)
            extend_left_time: Time in seconds to extend before the audio
            extend_right_time: Time in seconds to extend after the audio
        """
        ACEStepLatentUtils.validate_ace_latent_shape(source_latents)
        
        self.source_latents = source_latents
        self.left_frames = ACEStepLatentUtils.time_to_frame_index(extend_left_time)
        self.right_frames = ACEStepLatentUtils.time_to_frame_index(extend_right_time)
        
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