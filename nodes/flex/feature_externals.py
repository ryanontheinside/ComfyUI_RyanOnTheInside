import torch
import torch.nn.functional as F
import numpy  as np
from ... import RyanOnTheInside
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from ... import RyanOnTheInside
import cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import math

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





#TODO: sub somthing else
class DepthShapeModifier(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "mask": ("MASK",),
                # "feature": ("FEATURE",),
                # "feature_param": (["gradient_steepness", "depth_min", "depth_max", "strength"],),
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
            # Modify feature parameters based on the feature value
            # gradient_steepness, depth_min, depth_max, strength = self.modify_feature_param(
            #     feature, feature_param, gradient_steepness, depth_min, depth_max, strength
            # ) #TODO
            
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

    def modify_feature_param(self, feature, feature_param, gradient_steepness, depth_min, depth_max, strength):
        frames = feature.frame_count
        for i in range(frames):
            value = feature.get_value_at_frame(i)
            if feature_param == "gradient_steepness":
                gradient_steepness *= value
            elif feature_param == "depth_min":
                depth_min += value
            elif feature_param == "depth_max":
                depth_max -= value
            elif feature_param == "strength":
                strength *= value
        return gradient_steepness, depth_min, depth_max, strength
    

class DepthShapeModifierPrecise(FlexExternalModulator):
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
                "composite_method": (["linear","depth_aware", "add", "subtract", "multiply", "divide", "screen", "overlay", "protrude"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "modify_depth"
    CATEGORY = "RyanOnTheInside/DepthModifiers"

    def modify_depth(self, depth_map, mask, gradient_steepness, depth_min, depth_max, strength, composite_method):
        device = depth_map.device
        mask = mask.to(device).bool()
        
        b, h, w, c = depth_map.shape

        modified_depths = []
        for i in range(b):
            mask_i = mask[i].cpu().numpy().astype(np.uint8)

            # Compute the Euclidean distance transform
            distance = cv2.distanceTransform(mask_i, cv2.DIST_L2, 5)

            # Find local maxima in distance transform
            coordinates = peak_local_max(distance, min_distance=1, labels=mask_i)
            local_maxi = np.zeros_like(distance, dtype=bool)
            local_maxi[coordinates[:, 0], coordinates[:, 1]] = True

            markers = ndi.label(local_maxi)[0]

            # Apply the watershed algorithm to segment the particles
            labels = watershed(-distance, markers, mask=mask_i)

            sphere_gradients = torch.zeros((h, w), device=device)

            # Convert labels to torch tensor
            labels_torch = torch.from_numpy(labels).to(device)
            num_labels = labels_torch.max().item()

            for label in range(1, num_labels + 1):
                # Create a mask for this label
                component_mask = (labels_torch == label)
                ys, xs = component_mask.nonzero(as_tuple=True)
                if len(xs) == 0 or len(ys) == 0:
                    continue  # Skip if no pixels found

                center_x = torch.mean(xs.float())
                center_y = torch.mean(ys.float())

                # Calculate approximate radius
                radius = torch.sqrt(component_mask.sum().float() / math.pi)

                # Generate spherical gradient
                y_grid, x_grid = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                distances = torch.sqrt((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2)
                sphere_gradient = torch.clamp(1 - (distances / radius) ** gradient_steepness, 0, 1)

                # Apply gradient only to this component
                sphere_gradients += sphere_gradient * component_mask.float()

            # Scale gradient to depth range
            depth_gradient = depth_min + sphere_gradients * (depth_max - depth_min)

            # Store depth_min and depth_max as instance variables
            self.depth_min = depth_min
            self.depth_max = depth_max

            # Apply composite method
            modified_depth = self.apply_composite_method(depth_map[i], depth_gradient, mask[i], composite_method, strength)
            modified_depths.append(modified_depth)

        result = torch.stack(modified_depths, dim=0)
        return (result,)

    def apply_composite_method(self, original_depth, depth_gradient, mask, method, strength):
        mask = mask.unsqueeze(-1).repeat(1, 1, original_depth.shape[-1])
        depth_gradient = depth_gradient.unsqueeze(-1).repeat(1, 1, original_depth.shape[-1])
        
        if method == "protrude":
            # Calculate the difference between depth_gradient and original_depth
            difference = depth_gradient - original_depth
            
            # Only apply positive differences (protrusions)
            positive_difference = torch.clamp(difference, min=0)
            
            # Scale the protrusion by strength
            scaled_protrusion = positive_difference * strength
            
            # Add the scaled protrusion to the original depth
            protrusion = original_depth + scaled_protrusion
            
            # Create a mask for areas where original_depth is within the valid range
            valid_range_mask = (original_depth >= self.depth_min) & (original_depth <= self.depth_max)
            
            # Apply the protrusion only where the mask is active and original_depth is within the valid range
            return torch.where(mask & valid_range_mask, protrusion, original_depth)
        elif method == "linear":
            return original_depth * (1 - mask * strength) + depth_gradient * mask * strength
        elif method == "add":
            return torch.clamp(original_depth + depth_gradient * mask * strength, 0, 1)
        elif method == "subtract":
            return torch.clamp(original_depth - depth_gradient * mask * strength, 0, 1)
        elif method == "multiply":
            return original_depth * (1 + (depth_gradient - 1) * mask * strength)
        elif method == "divide":
            return original_depth / (1 + (1/depth_gradient - 1) * mask * strength)
        elif method == "screen":
            return 1 - (1 - original_depth) * (1 - depth_gradient * mask * strength)
        elif method == "overlay":
            condition = original_depth <= 0.5
            result = torch.where(
                condition,
                2 * original_depth * depth_gradient,
                1 - 2 * (1 - original_depth) * (1 - depth_gradient)
            )
            return original_depth * (1 - mask * strength) + result * mask * strength
        elif method == "depth_aware":
            # Only use depth_gradient where it's greater than original_depth (closer to camera)
            result = torch.where(depth_gradient > original_depth, depth_gradient, original_depth)
            return original_depth * (1 - mask * strength) + result * mask * strength
        else:
            raise ValueError(f"Unknown composite method: {method}")

    def modify_feature_param(self, feature, feature_param, gradient_steepness, depth_min, depth_max, strength):
        frames = feature.frame_count
        for i in range(frames):
            value = feature.get_value_at_frame(i)
            if feature_param == "gradient_steepness":
                gradient_steepness *= value
            elif feature_param == "depth_min":
                depth_min += value
            elif feature_param == "depth_max":
                depth_max -= value
            elif feature_param == "strength":
                strength *= value
        return gradient_steepness, depth_min, depth_max, strength
