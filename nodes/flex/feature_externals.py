import torch
import torch.nn.functional as F
import numpy  as np
from ... import RyanOnTheInside
from ... import RyanOnTheInside

class FlexExternalModulator(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexExternalMod"

class FeatureToWeightsStrategy(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature": ("FEATURE",),
            }
        }

    RETURN_TYPES = ("WEIGHTS_STRATEGY",)
    RETURN_NAMES = ("WEIGHTS_STRATEGY",)
    FUNCTION = "convert"

    def convert(self, feature):
        frames = feature.frame_count
        values = [feature.get_value_at_frame(i) for i in range(frames)]
        
        weights_str = ", ".join(map(lambda x: f"{x:.8f}", values))

        weights_strategy = {
            "weights": weights_str,
            "timing": "custom",
            "frames": frames,
            "start_frame": 0,
            "end_frame": frames,
            "add_starting_frames": 0,
            "add_ending_frames": 0,
            "method": "full batch",
            "frame_count": frames,
        }


        return (weights_strategy,)



import torch
import torch.nn.functional as F
import cv2

class DepthShapeModifier(RyanOnTheInside):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "mask": ("MASK",),
                "gradient_steepness": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "depth_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "modify_depth"
    CATEGORY = "RyanOnTheInside/DepthModifiers"

    def modify_depth(self, depth_map, mask, gradient_steepness, depth_min, depth_max, strength):
        device = depth_map.device
        mask = mask.to(device).bool()
        
        b, h, w, c = depth_map.shape
        
        modified_depths = []
        for i in range(b):
            mask_i = mask[i].cpu().numpy().astype(np.uint8)
            
            # Use cv2 to find contours (separate shapes)
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            sphere_gradients = torch.zeros((h, w), device=device)
            
            for contour in contours:
                # Create a mask for this contour
                component_mask = np.zeros_like(mask_i)
                cv2.drawContours(component_mask, [contour], 0, 1, -1)
                component_mask = torch.from_numpy(component_mask).to(device).bool()
                
                # Find center and radius of the circular component
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    continue  # Skip this contour if we can't find its center
                
                # Calculate radius (approximate)
                radius = np.sqrt(cv2.contourArea(contour) / np.pi)
                
                # Generate spherical gradient
                y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                distances = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
                sphere_gradient = torch.clamp(1 - (distances / radius) ** gradient_steepness, 0, 1)
                
                # Apply gradient only to this component
                sphere_gradients += sphere_gradient * component_mask.float()
            
            # Scale gradient to depth range
            depth_gradient = depth_min + sphere_gradients * (depth_max - depth_min)
            
            # Apply gradient to depth map
            modified_depth = depth_map[i].clone()
            depth_gradient = depth_gradient.unsqueeze(-1).repeat(1, 1, c)
            modified_depth = torch.where(mask[i].unsqueeze(-1).repeat(1, 1, c),
                                         depth_gradient,
                                         modified_depth)
            
            # Blend modified depth with original depth
            blend_mask = (mask[i].unsqueeze(-1) * strength).repeat(1, 1, c)
            modified_depth = depth_map[i] * (1 - blend_mask) + modified_depth * blend_mask
            modified_depths.append(modified_depth)
        
        result = torch.stack(modified_depths, dim=0)
        return (result,)