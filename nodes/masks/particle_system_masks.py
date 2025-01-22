import numpy as np
from .mask_base_particle_system import ParticleSystemMaskBase
from typing import List, Tuple
import cv2
from ... import RyanOnTheInside
from ...tooltips import apply_tooltips

@apply_tooltips
class ParticleEmissionMask(ParticleSystemMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "emission_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional":{
                "vortices": ("VORTEX",),                
                "wells": ("GRAVITY_WELL",),
                "well_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "static_bodies": ("STATIC_BODY",),
            }
        }

    RETURN_TYPES = ("MASK","IMAGE")
    FUNCTION = "main_function"

    
    def process_single_mask(self, mask: np.ndarray, frame_index: int, emission_strength: float, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        
        self.modulate_parameters(frame_index, mask)
        
        particle_mask = np.zeros_like(mask, dtype=np.float32)
        particle_image = np.zeros((*mask.shape, 3), dtype=np.float32)
        
        # Draw particles
        for particle in self.particles:
            particle_mask, particle_image = self.draw_particle(particle_mask, particle_image, particle)
        
        result_mask = np.maximum(mask, particle_mask * emission_strength)
        
        # The drawing of vortices and gravity wells will be handled in the base class
        
        result_image = np.maximum(particle_image, np.stack([mask] * 3, axis=-1))
        
        return result_mask, result_image

    def draw_mask_segments(self, shape):
        debug_mask = np.zeros((*shape, 3), dtype=np.float32)
        for segment in self.mask_shapes:
            p1 = tuple(map(int, segment.a))
            p2 = tuple(map(int, segment.b))
            cv2.line(debug_mask, p1, p2, (0, 0, 1), 2)  # Draw red lines with thickness 2
        return np.max(debug_mask, axis=2)

    def main_function(self, masks, strength, invert, subtract_original, grow_with_blur, emission_strength, static_bodies=None, **kwargs):
        super().initialize()
        return super().main_function(masks, strength, invert, subtract_original, grow_with_blur, 
                                     emission_strength=emission_strength, 
                                     static_bodies=static_bodies, **kwargs)    

class ParticleSystemModulatorBase(RyanOnTheInside):
    CATEGORY= f"{ParticleSystemMaskBase.CATEGORY}/Modulators"

@apply_tooltips
class EmitterModulationBase(ParticleSystemModulatorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "effect_duration": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "temporal_easing": (["ease_in_out", "linear", "bounce", "elastic", "none"],),
                "palindrome": ("BOOLEAN", {"default": False}),
                "random": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "previous_modulation": ("EMITTER_MODULATION",),
                "feature": ("FEATURE",),
            }
        }

    RETURN_TYPES = ("EMITTER_MODULATION",)
    FUNCTION = "create_modulation"
    CATEGORY = f"{ParticleSystemModulatorBase.CATEGORY}/Emitters"

    def create_modulation(self, start_frame, end_frame, effect_duration, temporal_easing, palindrome, random, previous_modulation=None, feature=None):
        modulation = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "effect_duration": effect_duration,
            "temporal_easing": temporal_easing,
            "palindrome": palindrome,
            "random": random,
            "feature": feature,
        }

        modulation_type = self.__class__.__name__

        if previous_modulation is None:
            modulation_chain = []
        else:
            modulation_chain = previous_modulation.copy()
            # Check if this type of modulation already exists in the chain
            if any(m.get("type") == modulation_type for m in modulation_chain):
                raise ValueError(f"A {modulation_type} already exists in the chain.")

        modulation["type"] = modulation_type
        modulation_chain.append(modulation)
        return (modulation_chain,)

@apply_tooltips
class EmitterEmissionRateModulation(EmitterModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "target_emission_rate": ("FLOAT", {"default": 50.0, "min": 0.1, "max": 100.0, "step": 0.1}),
            },
            "optional": super().INPUT_TYPES()["optional"],
        }

    FUNCTION = "create_emission_rate_modulation"

    def create_emission_rate_modulation(self, target_emission_rate, **kwargs):
        modulation_chain = super().create_modulation(**kwargs)[0]
        modulation_chain[-1]["target_emission_rate"] = target_emission_rate
        return (modulation_chain,)


@apply_tooltips
class Vortex(ParticleSystemModulatorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 10.0}),
                "radius": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 500.0, "step": 10.0}),
                "inward_factor": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "movement_speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "color": ("STRING", {"default": "(0,127,255)"}),
                "draw": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "previous_vortex": ("VORTEX",),
            }
        }

    RETURN_TYPES = ("VORTEX",)
    FUNCTION = "create_vortex"

    def create_vortex(self, x, y, strength, radius, inward_factor, movement_speed, color, draw, previous_vortex=None):
        vortex = {
            "x": x,
            "y": y,
            "strength": strength,
            "radius": radius,
            "inward_factor": inward_factor,
            "movement_speed": movement_speed,
            "color": color,
            "draw": draw,
        }
        
        if previous_vortex is None:
            vortex_list = [vortex]
        else:
            vortex_list = previous_vortex + [vortex]
        
        return (vortex_list,)

@apply_tooltips
class GravityWell(ParticleSystemModulatorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 5000.0, "step": 10.0}),
                "radius": ("FLOAT", {"default": 100.0, "min": 10.0, "max": 500.0, "step": 10.0}),
                "type": (["attract", "repel"],),
                "color": ("STRING", {"default": "(255,127,0)"}),
                "draw": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "previous_well": ("GRAVITY_WELL",),
            }
        }

    RETURN_TYPES = ("GRAVITY_WELL",)
    FUNCTION = "create_gravity_well"

    def create_gravity_well(self, x, y, strength, radius, type, color, draw, previous_well=None):
        well = {
            "x": x,
            "y": y,
            "strength": strength,
            "radius": radius,
            "type": type,
            "color": color,
            "draw": draw,
        }
        
        if previous_well is None:
            well_list = [well]
        else:
            well_list = previous_well + [well]
        
        return (well_list,)

@apply_tooltips
class ParticleEmitter(ParticleSystemModulatorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emitter_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emitter_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "particle_direction": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "particle_spread": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "particle_size": ("FLOAT", {"default": 17.4, "min": 1.0, "max": 400.0, "step": 0.1}),
                "particle_speed": ("FLOAT", {"default": 330.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "emission_rate": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "color": ("STRING", {"default": "(255,255,255)"}),
                "initial_plume": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "emission_radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
                "previous_emitter": ("PARTICLE_EMITTER",),
                "emitter_movement": ("EMITTER_MOVEMENT",),
                "spring_joint_setting": ("SPRING_JOINT_SETTING",),
                "particle_modulation":("PARTICLE_MODULATION",),
                "emitter_modulation": ("EMITTER_MODULATION",),
            }
        }

    RETURN_TYPES = ("PARTICLE_EMITTER",)
    FUNCTION = "create_emitter"

    def create_emitter(self, emitter_x, emitter_y, particle_direction, particle_spread, 
                       particle_size, particle_speed, emission_rate, color, initial_plume,
                       start_frame, end_frame, emission_radius, previous_emitter=None, 
                       emitter_movement=None, spring_joint_setting=None, particle_modulation=None,
                       emitter_modulation=None):
        emitter = {
            "emitter_x": emitter_x,
            "emitter_y": emitter_y,
            "particle_direction": particle_direction,
            "particle_spread": particle_spread,
            "particle_size": particle_size,
            "particle_speed": particle_speed,
            "emission_rate": emission_rate,
            "color": color,
            "initial_plume": initial_plume,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "emission_radius": emission_radius,
        }
        
        if emitter_movement:
            emitter["movement"] = emitter_movement
        
        if spring_joint_setting:
            emitter["spring_joint_setting"] = spring_joint_setting
        
        if particle_modulation:
            emitter["particle_modulation"]  = particle_modulation

        if emitter_modulation:
            emitter["emitter_modulation"] = emitter_modulation

        if previous_emitter is None:
            emitter_list = [emitter]
        else:
            emitter_list = previous_emitter + [emitter]
        
        return (emitter_list,)

@apply_tooltips
class SpringJointSetting(ParticleSystemModulatorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stiffness": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "damping": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "rest_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "max_distance": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 500.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("SPRING_JOINT_SETTING",)
    FUNCTION = "create_setting"

    def create_setting(self, stiffness, damping, rest_length, max_distance):
        return ({
            "stiffness": stiffness,
            "damping": damping,
            "rest_length": rest_length,
            "max_distance": max_distance,
        },)
    
@apply_tooltips
class EmitterMovement(ParticleSystemModulatorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emitter_x_frequency": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "emitter_x_amplitude": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "emitter_y_frequency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "emitter_y_amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.5, "step": 0.01}),
                "direction_frequency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "direction_amplitude": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "feature_param": (["emitter_x_frequency", "emitter_y_frequency", "direction_frequency"],),
            },
            "optional": {
                "feature": ("FEATURE",),
            }
        }

    RETURN_TYPES = ("EMITTER_MOVEMENT",)
    FUNCTION = "create_movement"
    CATEGORY = f"{ParticleSystemModulatorBase.CATEGORY}/EmitterModulators"
    def create_movement(self, emitter_x_frequency, emitter_y_frequency, direction_frequency,
                        emitter_x_amplitude, emitter_y_amplitude, direction_amplitude,
                        feature_param, feature=None):
        movement = {
            "emitter_x_frequency": emitter_x_frequency,
            "emitter_y_frequency": emitter_y_frequency,
            "direction_frequency": direction_frequency,
            "emitter_x_amplitude": emitter_x_amplitude,
            "emitter_y_amplitude": emitter_y_amplitude,
            "direction_amplitude": direction_amplitude,
            "feature_param": feature_param,
            "feature": feature,
        }
        return (movement,)

@apply_tooltips
class StaticBody(ParticleSystemModulatorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_type": (["segment", "polygon"],),
                "x1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "x2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
               # "radius": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
                "elasticity": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.1}),
                "friction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "draw": ("BOOLEAN", {"default": True}),
                "color": ("STRING", {"default": "(255,255,255)"}),
            },
            "optional": {
                "previous_body": ("STATIC_BODY",),
            }
        }

    RETURN_TYPES = ("STATIC_BODY",)
    FUNCTION = "create_static_body"

    def create_static_body(self, shape_type, x1, y1, x2, y2, elasticity, friction, draw, color, previous_body=None):
        body = {
            "shape_type": shape_type,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            # "radius": radius,
            "elasticity": elasticity,
            "friction": friction,
            "draw": draw,
            "color": color,
        }
        
        if previous_body is None:
            body_list = [body]
        else:
            body_list = previous_body + [body]
        
        return (body_list,)

@apply_tooltips
class ParticleModulationBase(ParticleSystemModulatorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "effect_duration": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "temporal_easing": (["ease_in_out", "linear", "bounce", "elastic", "none"],),
                "palindrome": ("BOOLEAN", {"default": False}),
                "random": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "previous_modulation": ("PARTICLE_MODULATION",),
                "feature": ("FEATURE",),
            }
        }
    
    CATEGORY = f"{ParticleSystemModulatorBase.CATEGORY}/ParticleModulators"
    
    def create_modulation(self, start_frame, end_frame, effect_duration, temporal_easing, palindrome, random, previous_modulation=None, feature=None):
        modulation = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "effect_duration": effect_duration,
            "temporal_easing": temporal_easing,
            "palindrome": palindrome,
            "random": random,
            "feature": feature,
        }
        
        modulation_type = self.__class__.__name__
        
        if previous_modulation is None:
            modulation_chain = []
        else:
            modulation_chain = previous_modulation.copy()
            # Check if this type of modulation already exists in the chain
            if any(m.get("type") == modulation_type for m in modulation_chain):
                raise ValueError(f"A {modulation_type} already exists in the chain.")
        
        modulation["type"] = modulation_type
        modulation_chain.append(modulation)
        return (modulation_chain,)

@apply_tooltips
class ParticleSizeModulation(ParticleModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "target_size": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 400.0, "step": 0.1}),
            },
            "optional": super().INPUT_TYPES()["optional"],
        }

    RETURN_TYPES = ("PARTICLE_MODULATION",)
    FUNCTION = "create_size_modulation"
    
    CATEGORY = f"{ParticleSystemModulatorBase.CATEGORY}/ParticleModulators"
    
    def create_size_modulation(self, target_size, **kwargs):
        modulation_chain = super().create_modulation(**kwargs)[0]
        modulation_chain[-1]["target_size"] = target_size
        return (modulation_chain,)

@apply_tooltips
class ParticleSpeedModulation(ParticleModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "target_speed": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
            },
            "optional": super().INPUT_TYPES()["optional"],
        }

    RETURN_TYPES = ("PARTICLE_MODULATION",)
    FUNCTION = "create_speed_modulation"
    CATEGORY = f"{ParticleSystemModulatorBase.CATEGORY}/ParticleModulators"

    def create_speed_modulation(self, target_speed, **kwargs):
        modulation_chain = super().create_modulation(**kwargs)[0]
        modulation_chain[-1]["target_speed"] = target_speed
        return (modulation_chain,)

@apply_tooltips
class ParticleColorModulation(ParticleModulationBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "target_color": ("STRING", {"default": "(255,255,255)"}),
            },
            "optional": super().INPUT_TYPES()["optional"],
        }

    RETURN_TYPES = ("PARTICLE_MODULATION",)
    FUNCTION = "create_color_modulation"
    CATEGORY = f"{ParticleSystemModulatorBase.CATEGORY}/ParticleModulators"
    

    def create_color_modulation(self, target_color, **kwargs):
        modulation_chain = super().create_modulation(**kwargs)[0]
        modulation_chain[-1]["target_color"] = target_color
        return (modulation_chain,)
