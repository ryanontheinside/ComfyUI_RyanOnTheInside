import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import folder_paths
import comfy.utils
import time
import copy
import dill
import yaml
from ultralytics import YOLO
import importlib.util
from .flex_externals import FlexExternalModulator

#TODO / NOTE: this more robust  import  module from path should logically be moved to the FlexExternals baseclass.

#NOTE: attempt to import advanced live portrait
def import_module_from_path(module_name, module_path):


    package_dir = os.path.dirname(module_path)
    original_sys_path = sys.path.copy()
    
    try:
        # Add the parent directory to sys.path
        parent_dir = os.path.dirname(package_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        # Add LivePortrait directory to sys.path
        liveportrait_path = os.path.join(package_dir, 'LivePortrait')
        if os.path.exists(liveportrait_path) and liveportrait_path not in sys.path:
            sys.path.insert(0, liveportrait_path)
            
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
        
        # Create package spec
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            os.path.join(package_dir, '__init__.py')
        )
        
        # Create LivePortrait package spec
        liveportrait_spec = importlib.util.spec_from_file_location(
            f"{package_name}.LivePortrait",
            os.path.join(liveportrait_path, '__init__.py')
        )
        
        # Add package to sys.modules with proper specs
        if package_name not in sys.modules:
            package = type(
                'Package',
                (),
                {
                    '__path__': [package_dir],
                    '__package__': package_name,
                    '__spec__': package_spec,
                    'LivePortrait': type(
                        'LivePortrait',
                        (),
                        {
                            '__path__': [liveportrait_path],
                            '__package__': f"{package_name}.LivePortrait",
                            '__spec__': liveportrait_spec
                        }
                    )
                }
            )
            sys.modules[package_name] = package
            sys.modules[f"{package_name}.LivePortrait"] = package.LivePortrait
        
        # Execute the module
        spec.loader.exec_module(module)
        
        return module
        
    finally:
        # Restore original sys.path
        sys.path = original_sys_path

#NOTE: need init files in AdvancedLivePortrait

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
custom_nodes_dir = os.path.abspath(os.path.join(current_directory, '..', '..', '..'))

# Path to AdvancedLivePortrait's nodes.py
advanced_live_portrait_dir = os.path.join(custom_nodes_dir, 'ComfyUI-AdvancedLivePortrait')

# Import the nodes.py module
advanced_nodes_path = os.path.join(advanced_live_portrait_dir, 'nodes.py')
advanced_nodes_module = import_module_from_path('advanced_live_portrait_nodes', advanced_nodes_path)

# Import required classes and variables from advanced_nodes_module
ExpressionEditor = advanced_nodes_module.ExpressionEditor
g_engine = advanced_nodes_module.g_engine
ExpressionSet = advanced_nodes_module.ExpressionSet
retargeting = advanced_nodes_module.retargeting

# Import other necessary components similarly
get_rotation_matrix = advanced_nodes_module.get_rotation_matrix
get_rgb_size = advanced_nodes_module.get_rgb_size
pil2tensor = advanced_nodes_module.pil2tensor

class FlexExpressionEditor(ExpressionEditor):
    """
    Enhanced version of ExpressionEditor that adds support for feature-based parameter modulation.
    Allows for dynamic modification of facial expressions based on input features (e.g. audio, motion, etc.).
    
    TODO: Consider refactoring these methods into the base ExpressionEditor class:
    - generate_preview_image
    - process_sample_image
    - create_expression_set
    
    These methods have been implemented here to avoid modifying the original ExpressionEditor
    until the author can review the changes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "constrain_min_max": ("BOOLEAN", {"default": True})
        })
        # Rename motion_link to flex_motion_link in optional inputs
        base_inputs["optional"]["flex_motion_link"] = base_inputs["optional"].pop("motion_link")
        base_inputs["optional"].update({
            "feature": ("FEATURE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "feature_param": (cls.get_modifiable_params(), {"default": cls.get_modifiable_params()[0]}),
            "feature_mode": (["relative", "absolute"], {"default": "absolute"}),
             
        })
        
        return base_inputs

    RETURN_TYPES = ("IMAGE", "EDITOR_LINK", "EXP_DATA", "STRING")
    RETURN_NAMES = ("image", "flex_motion_link", "save_exp", "command")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = f"{FlexExternalModulator.CATEGORY}/Advanced-Live-Portrait"
    @classmethod
    def get_modifiable_params(cls):


        return ["rotate_pitch", "rotate_yaw", "rotate_roll", "blink", "eyebrow", "wink", "pupil_x", "pupil_y",
                "aaa", "eee", "woo", "smile", "None"]

    def modulate_param(self, base_value, feature_value, strength, mode="relative", param_name=None, constrain_min_max=True):
        """
        Helper method to modulate a parameter based on feature value.
        
        Args:
            base_value: Original parameter value
            feature_value: Feature value to modulate with
            strength: Modulation strength
            mode: Either "relative" or "absolute"
            param_name: Name of the parameter being modulated (for min/max constraints)
            constrain_min_max: Whether to constrain the output to the parameter's min/max values
            
        Returns:
            Modulated parameter value
        """
        if mode == "relative":
            modulated_value = base_value * (1 + feature_value * strength)
        else:  # absolute
            modulated_value = base_value * feature_value * strength

        if constrain_min_max and param_name:
            # Get parameter constraints from INPUT_TYPES
            param_info = self.INPUT_TYPES()["required"].get(param_name)
            if param_info and isinstance(param_info[1], dict):
                param_min = param_info[1].get("min")
                param_max = param_info[1].get("max")
                if param_min is not None:
                    modulated_value = max(param_min, modulated_value)
                if param_max is not None:
                    modulated_value = min(param_max, modulated_value)

        return modulated_value

    #TODO: base class
    def generate_preview_image(self, psi, preview_es):
        """
        Generate preview image from the expression set and PSI data.
        
        Args:
            psi: PSI data containing source image information
            preview_es: Expression set to apply for preview
            
        Returns:
            tuple: (output_image_tensor, results_dict)
        """
        s_info = psi.x_s_info
        pipeline = g_engine.get_pipeline()
        
        new_rotate = get_rotation_matrix(s_info['pitch'] + preview_es.r[0], 
                                       s_info['yaw'] + preview_es.r[1], 
                                       s_info['roll'] + preview_es.r[2])
        x_d_new = (s_info['scale'] * (1 + preview_es.s)) * ((s_info['kp'] + preview_es.e) @ new_rotate) + s_info['t']

        x_d_new = pipeline.stitching(psi.x_s_user, x_d_new)
        crop_out = pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
        crop_out = pipeline.parse_output(crop_out['out'])[0]
        crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
        out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)
        out_img = pil2tensor(out)

        filename = g_engine.get_temp_img_name()
        folder_paths.get_save_image_path(filename, folder_paths.get_temp_directory())
        img = Image.fromarray(crop_out)
        img.save(os.path.join(folder_paths.get_temp_directory(), filename), compress_level=1)
        results = [{"filename": filename, "type": "temp"}]
        
        return out_img, results

    #TODO: base class
    def initialize_psi(self, src_image, motion_link, crop_factor):
        """Initialize PSI data from source image or motion link"""
        new_editor_link = []
        
        if motion_link is not None:
            self.psi = motion_link[0]
            new_editor_link.append(self.psi)
        elif src_image is not None:
            if id(src_image) != id(self.src_image) or self.crop_factor != crop_factor:
                self.crop_factor = crop_factor
                self.psi = g_engine.prepare_source(src_image, crop_factor)
                self.src_image = src_image
            new_editor_link.append(self.psi)
        else:
            return None
            
        return new_editor_link

    #TODO: base class   
    def process_sample_image(self, sample_image, sample_parts, sample_ratio, es, rotate_pitch, rotate_yaw, rotate_roll):
        """Process sample image and apply transformations"""
        pipeline = g_engine.get_pipeline()
        
        if id(self.sample_image) != id(sample_image):
            self.sample_image = sample_image
            d_image_np = (sample_image * 255).byte().numpy()
            d_face = g_engine.crop_face(d_image_np[0], 1.7)
            i_d = g_engine.prepare_src_image(d_face)
            self.d_info = pipeline.get_kp_info(i_d)
            self.d_info['exp'][0, 5, 0] = 0
            self.d_info['exp'][0, 5, 1] = 0

        if sample_parts == "OnlyExpression" or sample_parts == "All":
            es.e += self.d_info['exp'] * sample_ratio
        if sample_parts == "OnlyRotation" or sample_parts == "All":
            rotate_pitch += self.d_info['pitch'] * sample_ratio
            rotate_yaw += self.d_info['yaw'] * sample_ratio
            rotate_roll += self.d_info['roll'] * sample_ratio
        elif sample_parts == "OnlyMouth":
            retargeting(es.e, self.d_info['exp'], sample_ratio, (14, 17, 19, 20))
        elif sample_parts == "OnlyEyes":
            retargeting(es.e, self.d_info['exp'], sample_ratio, (1, 2, 11, 13, 15, 16))
            
        return rotate_pitch, rotate_yaw, rotate_roll

    def run(self, rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
            src_ratio, sample_ratio, sample_parts, crop_factor, constrain_min_max, src_image=None, sample_image=None, 
            flex_motion_link=None, add_exp=None, feature=None, strength=1.0, feature_threshold=0.0, 
            feature_param="None", feature_mode="relative"):
        rotate_yaw = -rotate_yaw

        # Initialize PSI data
        new_editor_link = self.initialize_psi(src_image, flex_motion_link, crop_factor)
        if new_editor_link is None:
            return (None, None, None, None)

        pipeline = g_engine.get_pipeline()
        psi = self.psi
        s_info = psi.x_s_info

        es = ExpressionSet()

        # Process sample image if provided
        if sample_image is not None:
            rotate_pitch, rotate_yaw, rotate_roll = self.process_sample_image(
                sample_image, sample_parts, sample_ratio, es, rotate_pitch, rotate_yaw, rotate_roll
            )

        # Add any additional expression data
        if add_exp is not None:
            es.add(add_exp)

        # Prepare parameters dictionary
        params = {
            'blink': blink,
            'eyebrow': eyebrow,
            'wink': wink,
            'pupil_x': pupil_x,
            'pupil_y': pupil_y,
            'aaa': aaa,
            'eee': eee,
            'woo': woo,
            'smile': smile,
            'rotate_pitch': rotate_pitch,
            'rotate_yaw': rotate_yaw,
            'rotate_roll': rotate_roll,
        }

        # When calling calc_fe, unpack the parameters in the correct order
        def apply_params(es, params):
            return g_engine.calc_fe(
                es,
                params['blink'],
                params['eyebrow'],
                params['wink'],
                params['pupil_x'],
                params['pupil_y'],
                params['aaa'],
                params['eee'],
                params['woo'],
                params['smile'],
                params['rotate_pitch'],
                params['rotate_yaw'],
                params['rotate_roll']
            )

        # Check if feature is provided for modulation
        if (feature is not None) and feature_param != "None" and feature_param in params:
            total_frames = feature.frame_count
            feature_values = [feature.get_value_at_frame(i) for i in range(total_frames)]
        else:
            # No modulation
            total_frames = 1
            if flex_motion_link is not None and len(flex_motion_link) > 1:
                total_frames = len(flex_motion_link) - 1  # Subtract 1 because first element is psi
            feature_values = [1.0] * total_frames

        # Generate 'command' string expected by AdvancedLivePortrait node
        command_lines = []
        for idx in range(total_frames):
            # Start with base expression set or previous modifications
            if flex_motion_link is not None and idx + 1 < len(flex_motion_link):
                es_frame = ExpressionSet(es=flex_motion_link[idx + 1])
                if add_exp is not None:
                    es_frame.add(add_exp)
            else:
                es_frame = ExpressionSet(es=es)

            # Apply feature modulation if available
            frame_params = params.copy()
            if (feature is not None) and feature_param != "None" and feature_param in params:
                feature_value = feature_values[idx]
                if abs(feature_value) >= feature_threshold:
                    frame_params[feature_param] = self.modulate_param(
                        params[feature_param], 
                        feature_value, 
                        strength, 
                        feature_mode,
                        feature_param,
                        constrain_min_max
                    )

            # Calculate the new rotations and add them to existing ones
            new_rotations = apply_params(es_frame.e, frame_params)
            if flex_motion_link is not None and idx + 1 < len(flex_motion_link):
                es_frame.r += new_rotations  # Add to existing rotations
            else:
                es_frame.r = new_rotations  # Set new rotations
                
            new_editor_link.append(es_frame)

            # Create command line
            command_lines.append(f"{idx}=1:0")

        command = '\n'.join(command_lines)

        # Apply the expressions for preview image (using first frame)
        preview_es = new_editor_link[1] if len(new_editor_link) > 1 else es
        out_img, results = self.generate_preview_image(psi, preview_es)

        # Find the frame with maximum feature value and use its expression
        if feature is not None and feature_param != "None" and len(new_editor_link) > 1:
            max_idx = feature_values.index(max(feature_values))
            save_exp = new_editor_link[max_idx + 1]  # +1 because first element is psi
        else:
            save_exp = es

        return {"ui": {"images": results}, "result": (out_img, new_editor_link, save_exp, command)}