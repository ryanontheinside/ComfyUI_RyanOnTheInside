import torch
import torch.nn.functional as F
import numpy  as np
from ... import RyanOnTheInside
import os
import sys
import importlib.util
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
from ...tooltips import apply_tooltips
from ... import ProgressMixin


class FlexExternalModulator(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/ExternalTargets"

    @staticmethod
    def import_module_from_path(module_name, module_path):
        """Helper method to import external modules from other custom node packs.
        
        Args:
            module_name: Name to give the imported module
            module_path: Path to the module file
            
        Returns:
            The imported module
        """
        package_dir = os.path.dirname(module_path)
        original_sys_path = sys.path.copy()
        
        try:
            # Add the parent directory to sys.path
            parent_dir = os.path.dirname(package_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
                
            # Create module spec
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None:
                raise ImportError(f"Cannot create a module spec for {module_path}")
                
            module = importlib.util.module_from_spec(spec)
            
            # Add the module to sys.modules
            sys.modules[module_name] = module
            
            # Set up package structure
            package_name = os.path.basename(package_dir).replace('-', '_')
            module.__package__ = package_name
            
            # Execute the module
            spec.loader.exec_module(module)
            
            return module
            
        finally:
            # Restore original sys.path
            sys.path = original_sys_path

@apply_tooltips
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

_spline_category = f"{FlexExternalModulator.CATEGORY}/Spline"

@apply_tooltips
class FeatureToSplineData(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "mask_width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "mask_height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "sampling_method": (
                    [
                        'path',
                        'time',
                        'controlpoints'
                    ],
                    {
                        "default": 'time'
                    }
                ),
                "interpolation": (
                    [
                        'cardinal',
                        'monotone',
                        'basis',
                        'linear',
                        'step-before',
                        'step-after',
                        'polar',
                        'polar-reverse',
                    ],
                    {
                        "default": 'cardinal'
                    }
                ),
                "tension": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_output": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "float_output_type": (
                    [
                        'list',
                        'pandas series',
                        'tensor',
                    ],
                    {
                        "default": 'list'
                    }
                ),
            },
            "optional": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("mask", "coord_str", "float", "count", "normalized_str",)
    FUNCTION = "convert"
    CATEGORY = _spline_category

    def convert(self, feature, mask_width, mask_height, sampling_method, interpolation,
                tension, repeat_output, float_output_type, min_value=0.0, max_value=1.0):
        import torch

        import numpy as np
        import json

        
        # Retrieve values from the feature
        frames = feature.frame_count
        values = [feature.get_value_at_frame(i) for i in range(frames)]

        # Normalize feature values between min_value and max_value
        normalized_y_values = [min_value + (v * (max_value - min_value)) for v in values]

        # Set points_to_sample to frames
        points_to_sample = frames

        # Prepare x-coordinates scaled to the mask width
        x_values = np.linspace(0, mask_width - 1, frames)

        # Prepare y-values scaled to the mask height
        y_values = [(1.0 - v) * (mask_height - 1) for v in normalized_y_values]

        # Prepare output float based on the selected type
        if float_output_type == 'list':
            out_floats = normalized_y_values * repeat_output
        elif float_output_type == 'pandas series':
            try:
                import pandas as pd
            except ImportError:
                raise Exception("FeatureToSplineData: pandas is not installed. Please install pandas to use this output_type.")
            out_floats = pd.Series(normalized_y_values * repeat_output)
        elif float_output_type == 'tensor':
            out_floats = torch.tensor(normalized_y_values * repeat_output, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown float_output_type: {float_output_type}")

        # Create masks based on normalized y-values
        mask_tensors = []
        for y in normalized_y_values:
            # Create a grayscale image with intensity y
            mask = torch.full((mask_height, mask_width, 3), y, dtype=torch.float32)
            mask_tensors.append(mask)

        # Stack and process mask tensors
        masks_out = torch.stack(mask_tensors)
        masks_out = masks_out.repeat(repeat_output, 1, 1, 1)
        masks_out = masks_out.mean(dim=-1)

        # Prepare coordinate strings with coordinates scaled to mask dimensions
        coordinates = [{'x': float(x), 'y': float(y)} for x, y in zip(x_values, y_values)]
        coord_str = json.dumps(coordinates)
        normalized_str = coord_str
        count = len(out_floats)

        return (masks_out, coord_str, out_floats, count, normalized_str)

import torch
import numpy as np
import json
from scipy.interpolate import interp1d

@apply_tooltips
class SplineFeatureModulator(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING", {"multiline": False}),
                "feature": ("FEATURE",),
                "mask_width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "mask_height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "min_speed": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "float_output_type": (["list", "pandas series", "tensor"], {"default": 'list'}),
            },
            "optional": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("mask", "coord_str", "float", "count", "normalized_str",)
    FUNCTION = "modulate_spline"
    CATEGORY = _spline_category

    def modulate_spline(self, coordinates, feature, mask_width, mask_height, 
                       min_speed, max_speed, float_output_type,
                       min_value=0.0, max_value=1.0):
        import torch
        import numpy as np
        import json
        from scipy.interpolate import interp1d

        # Parse coordinates from JSON string
        coordinates = json.loads(coordinates)
        control_points = np.array([[point['x'], point['y']] for point in coordinates])

        # Get feature values and frames
        frames = feature.frame_count
        feature_values = np.array([feature.get_value_at_frame(i) for i in range(frames)])

        # Calculate speeds from feature values
        # Positive features: min_speed to max_speed forward
        # Negative features: min_speed to max_speed backward
        speeds = np.where(
            feature_values >= 0,
            min_speed + (feature_values * (max_speed - min_speed)),  # Forward speeds
            -(min_speed + (abs(feature_values) * (max_speed - min_speed)))  # Backward speeds
        )

        # Calculate cumulative positions along the path
        cumulative_movement = np.cumsum(speeds)
        # Normalize to [0, 1] range for path interpolation
        total_movement = np.abs(cumulative_movement[-1])
        if total_movement > 0:
            path_positions = (cumulative_movement / total_movement) % 1.0
        else:
            path_positions = np.zeros_like(cumulative_movement)

        # Interpolate control points
        t_orig = np.linspace(0, 1, len(control_points))
        x_interp = interp1d(t_orig, control_points[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
        y_interp = interp1d(t_orig, control_points[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')

        # Sample points along the path
        sampled_x = x_interp(path_positions)
        sampled_y = y_interp(path_positions)

        # Normalize y-values between min_value and max_value
        normalized_y_values = min_value + (sampled_y / (mask_height - 1)) * (max_value - min_value)

        # Prepare output float based on the selected type
        if float_output_type == 'list':
            out_floats = normalized_y_values.tolist()
        elif float_output_type == 'pandas series':
            try:
                import pandas as pd
            except ImportError:
                raise Exception("SplineFeatureModulator: pandas is not installed.")
            out_floats = pd.Series(normalized_y_values)
        elif float_output_type == 'tensor':
            out_floats = torch.tensor(normalized_y_values, dtype=torch.float32)

        # Create masks based on normalized y-values
        mask_tensors = []
        for y in normalized_y_values:
            mask = torch.full((mask_height, mask_width, 3), y, dtype=torch.float32)
            mask_tensors.append(mask)

        # Stack and process mask tensors
        masks_out = torch.stack(mask_tensors)
        masks_out = masks_out.mean(dim=-1)

        # Prepare output coordinates
        sampled_coordinates = [{'x': float(x), 'y': float(y)} for x, y in zip(sampled_x, sampled_y)]
        coord_str = json.dumps(sampled_coordinates)

        return (masks_out, coord_str, out_floats, len(out_floats), coord_str)

@apply_tooltips
class SplineRhythmModulator(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING", {"multiline": False}),
                "feature": ("FEATURE",),
                "mask_width": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "mask_height": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "direction": (["forward", "backward", "bounce"], {"default": "bounce"}),
                "float_output_type": (["list", "pandas series", "tensor"], {"default": 'list'}),
            },
            "optional": {
                "min_value": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -10000.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING", "FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("mask", "coord_str", "float", "count", "normalized_str",)
    FUNCTION = "modulate_rhythm"
    CATEGORY = _spline_category
    def modulate_rhythm(self, coordinates, feature, mask_width, mask_height, 
                       smoothing, direction, float_output_type,
                       min_value=0.0, max_value=1.0):
        print("\n=== Starting SplineRhythmModulator.modulate_rhythm ===")
        print(f"Input parameters: width={mask_width}, height={mask_height}")
        print(f"Smoothing: {smoothing}, Direction: {direction}")
        print(f"Value range: {min_value} to {max_value}")

        import torch
        import numpy as np
        import json
        from scipy.interpolate import interp1d
        from scipy.ndimage import gaussian_filter1d

        # Parse coordinates
        print("\nParsing coordinates...")
        try:
            coordinates = json.loads(coordinates)
            control_points = np.array([[point['x'], point['y']] for point in coordinates])
            print(f"Number of control points: {len(control_points)}")
            print(f"First point: {control_points[0]}, Last point: {control_points[-1]}")
        except Exception as e:
            print(f"Error parsing coordinates: {e}")
            raise

        # Get feature values
        print("\nExtracting feature values...")
        try:
            frames = feature.frame_count
            feature_values = np.array([feature.get_value_at_frame(i) for i in range(frames)])
            print(f"Number of frames: {frames}")
            print(f"Feature value range: {feature_values.min():.3f} to {feature_values.max():.3f}")
        except Exception as e:
            print(f"Error extracting feature values: {e}")
            raise

        # Apply smoothing
        print("\nApplying smoothing...")
        if smoothing > 0:
            try:
                sigma = smoothing * 2
                feature_values = gaussian_filter1d(feature_values, sigma)
                print(f"Smoothed value range: {feature_values.min():.3f} to {feature_values.max():.3f}")
            except Exception as e:
                print(f"Error in smoothing: {e}")
                raise

        # Normalize feature values
        print("\nNormalizing feature values...")
        try:
            feature_range = feature_values.max() - feature_values.min()
            if feature_range == 0:
                print("Warning: Feature values have zero range!")
                feature_values = np.zeros_like(feature_values)
            else:
                feature_values = (feature_values - feature_values.min()) / feature_range
            print(f"Normalized value range: {feature_values.min():.3f} to {feature_values.max():.3f}")
        except Exception as e:
            print(f"Error in normalization: {e}")
            raise

        # Convert to path positions
        print("\nConverting to path positions...")
        try:
            if direction == "forward":
                path_positions = feature_values
            elif direction == "backward":
                path_positions = 1 - feature_values
            else:  # bounce
                path_positions = np.abs(2 * feature_values - 1)
            print(f"Path positions range: {path_positions.min():.3f} to {path_positions.max():.3f}")
        except Exception as e:
            print(f"Error in path position conversion: {e}")
            raise

        # Interpolate points
        print("\nInterpolating points...")
        try:
            t_orig = np.linspace(0, 1, len(control_points))
            x_interp = interp1d(t_orig, control_points[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
            y_interp = interp1d(t_orig, control_points[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
            
            sampled_x = x_interp(path_positions)
            sampled_y = y_interp(path_positions)
            print(f"Sampled points range: x({sampled_x.min():.1f} to {sampled_x.max():.1f}), y({sampled_y.min():.1f} to {sampled_y.max():.1f})")
        except Exception as e:
            print(f"Error in interpolation: {e}")
            raise

        # Normalize y-values
        print("\nNormalizing y-values...")
        try:
            normalized_y_values = min_value + (sampled_y / (mask_height - 1)) * (max_value - min_value)
            print(f"Normalized y-values range: {normalized_y_values.min():.3f} to {normalized_y_values.max():.3f}")
        except Exception as e:
            print(f"Error in y-value normalization: {e}")
            raise

        # Prepare output float
        print("\nPreparing output float...")
        try:
            if float_output_type == 'list':
                out_floats = normalized_y_values.tolist()
                print(f"Output type: list, length: {len(out_floats)}")
            elif float_output_type == 'pandas series':
                import pandas as pd
                out_floats = pd.Series(normalized_y_values)
                print(f"Output type: pandas series, length: {len(out_floats)}")
            elif float_output_type == 'tensor':
                out_floats = torch.tensor(normalized_y_values, dtype=torch.float32)
                print(f"Output type: tensor, shape: {out_floats.shape}")
        except Exception as e:
            print(f"Error preparing output float: {e}")
            raise

        # Create masks
        print("\nCreating masks...")
        try:
            mask_tensors = []
            for y in normalized_y_values:
                mask = torch.full((mask_height, mask_width, 3), y, dtype=torch.float32)
                mask_tensors.append(mask)
            masks_out = torch.stack(mask_tensors)
            masks_out = masks_out.mean(dim=-1)
            print(f"Mask tensor shape: {masks_out.shape}")
        except Exception as e:
            print(f"Error creating masks: {e}")
            raise

        # Prepare output coordinates
        sampled_coordinates = [{'x': float(x), 'y': float(y)} for x, y in zip(sampled_x, sampled_y)]
        coord_str = json.dumps(sampled_coordinates)

        return (masks_out, coord_str, out_floats, len(out_floats), coord_str)

@apply_tooltips
class FeatureToFloat(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
            }
        }   
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_audio_weights"


    def get_audio_weights(self, feature):
        data = []
        for i in range(feature.frame_count):
            data.append(feature.get_value_at_frame(i))
        return (data,) 

@apply_tooltips
class FeatureToFilteredList(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "filter_type": (["peaks", "troughs", "above_threshold", "below_threshold", "significant_changes"],),
                "threshold_type": (["absolute", "relative", "adaptive"],),
                "threshold_value": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_distance": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("filtered_indices", "filtered_count", "filtered_binary", "filtered_indices_str",)
    FUNCTION = "filter_feature"

    def filter_feature(self, feature, filter_type, threshold_type, threshold_value, smoothing, min_distance):
        import numpy as np
        from scipy.signal import find_peaks, find_peaks_cwt
        from scipy.ndimage import gaussian_filter1d

        # Get feature values
        values = np.array([feature.get_value_at_frame(i) for i in range(feature.frame_count)])
        
        # Apply smoothing if needed
        if smoothing > 0:
            values = gaussian_filter1d(values, sigma=smoothing * 10)
        
        # Normalize values to [0, 1] range
        if values.max() != values.min():
            normalized_values = (values - values.min()) / (values.max() - values.min())
        else:
            normalized_values = values

        # Determine threshold based on threshold_type
        if threshold_type == "absolute":
            threshold = threshold_value
        elif threshold_type == "relative":
            threshold = np.percentile(normalized_values, threshold_value * 100)
        else:  # adaptive
            # Use standard deviation for adaptive thresholding
            threshold = np.mean(normalized_values) + threshold_value * np.std(normalized_values)

        # Apply filtering based on filter_type
        if filter_type == "peaks":
            peaks, _ = find_peaks(normalized_values, height=threshold, distance=min_distance)
            filtered_indices = peaks.tolist()
        elif filter_type == "troughs":
            # Invert the signal to find troughs
            inverted_values = 1 - normalized_values
            troughs, _ = find_peaks(inverted_values, height=1-threshold, distance=min_distance)
            filtered_indices = troughs.tolist()
        elif filter_type == "above_threshold":
            filtered_indices = np.where(normalized_values > threshold)[0].tolist()
        elif filter_type == "below_threshold":
            filtered_indices = np.where(normalized_values < threshold)[0].tolist()
        else:  # significant_changes
            # Calculate differences between consecutive values
            differences = np.abs(np.diff(normalized_values))
            # Find points where the change is significant
            significant_changes = np.where(differences > threshold)[0]
            # Add 1 to indices since diff reduces array length by 1
            filtered_indices = (significant_changes + 1).tolist()

        # Create binary mask
        binary_mask = np.zeros_like(normalized_values)
        binary_mask[filtered_indices] = 1

        # Create comma-separated string of indices
        filtered_indices_str = ",".join(map(str, filtered_indices))

        return (filtered_indices, len(filtered_indices), binary_mask.tolist(), filtered_indices_str)

#TODO: sub somthing logical
_depth_category = "RyanOnTheInside/DepthModifiers"
@apply_tooltips
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
    CATEGORY = _depth_category

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
    


@apply_tooltips
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
    CATEGORY = _depth_category

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
    

@apply_tooltips
class FeatureToMask(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"

    def convert(self, feature):
        # Get dimensions from feature
        height = feature.height
        width = feature.width
        
        # Get normalized data as numpy array
        normalized_data = feature.get_normalized_data()
        if normalized_data is None:
            # If no data available, use frame-by-frame normalization
            normalized_data = np.array([feature.get_value_at_frame(i) for i in range(feature.frame_count)])
            if len(normalized_data) > 0:
                min_val = np.min(normalized_data)
                max_val = np.max(normalized_data)
                if max_val > min_val:
                    normalized_data = (normalized_data - min_val) / (max_val - min_val)
                else:
                    normalized_data = np.zeros_like(normalized_data)
        
        # Convert to torch tensor and reshape for broadcasting
        normalized_tensor = torch.from_numpy(normalized_data).float()
        
        # Create masks for all frames at once by expanding the normalized data
        masks_out = normalized_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, height, width)
        return (masks_out,)
    
