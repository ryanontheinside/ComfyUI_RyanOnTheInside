import numpy as np
from .mask_base import ParticleSystemMaskBase
from typing import List, Tuple
import cv2
    
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
    CATEGORY = "Masks/Particle Systems"

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
class Vortex:
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
    CATEGORY = "RyanOnTheInside/ParticleSystems"

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

class GravityWell:
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
    CATEGORY = "RyanOnTheInside/ParticleSystems"

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

class ParticleEmitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emitter_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emitter_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "particle_direction": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "particle_spread": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "particle_size": ("FLOAT", {"default": 17.4, "min": 1.0, "max": 100.0, "step": 0.1}),
                "particle_speed": ("FLOAT", {"default": 330.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "emission_rate": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "color": ("STRING", {"default": "(255,255,255)"}),
                "initial_plume": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "previous_emitter": ("PARTICLE_EMITTER",),
                "emitter_movement": ("EMITTER_MOVEMENT",),
                "spring_joint_setting": ("SPRING_JOINT_SETTING",),
            }
        }

    RETURN_TYPES = ("PARTICLE_EMITTER",)
    FUNCTION = "create_emitter"
    CATEGORY = "RyanOnTheInside/ParticleSystems"

    def create_emitter(self, emitter_x, emitter_y, particle_direction, particle_spread, 
                       particle_size, particle_speed, emission_rate, color, initial_plume,
                       previous_emitter=None, emitter_movement=None, spring_joint_setting=None):
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
        }
        
        if emitter_movement:
            emitter["movement"] = emitter_movement
        
        if spring_joint_setting:
            emitter["spring_joint_setting"] = spring_joint_setting
        
        if previous_emitter is None:
            emitter_list = [emitter]
        else:
            emitter_list = previous_emitter + [emitter]
        
        return (emitter_list,)

class SpringJointSetting:
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
    CATEGORY = "RyanOnTheInside/ParticleSystems"

    def create_setting(self, stiffness, damping, rest_length, max_distance):
        return ({
            "stiffness": stiffness,
            "damping": damping,
            "rest_length": rest_length,
            "max_distance": max_distance,
        },)
    
class EmitterMovement:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emitter_x_frequency": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "emitter_x_amplitude": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "emitter_y_frequency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "emitter_y_amplitude": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.5, "step": 0.01}),
                "direction_frequency": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "direction_amplitude": ("FLOAT", {"default": 360.0, "min": 0.0, "max": 180.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("EMITTER_MOVEMENT",)
    FUNCTION = "create_movement"
    CATEGORY = "RyanOnTheInside/ParticleSystems"

    def create_movement(self, emitter_x_frequency, emitter_y_frequency, direction_frequency,
                        emitter_x_amplitude, emitter_y_amplitude, direction_amplitude):
        movement = {
            "emitter_x_frequency": emitter_x_frequency,
            "emitter_y_frequency": emitter_y_frequency,
            "direction_frequency": direction_frequency,
            "emitter_x_amplitude": emitter_x_amplitude,
            "emitter_y_amplitude": emitter_y_amplitude,
            "direction_amplitude": direction_amplitude,
        }
        return (movement,)

class StaticBody:
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
    CATEGORY = "RyanOnTheInside/ParticleSystems"

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


# class PulsatingParticleSystemMask(ParticleSystemMaskBase):
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 **super().INPUT_TYPES()["required"],
#                 "size_frequency": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
#                 "size_amplitude": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}),
#                 "emission_frequency": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01}),
#                 "emission_amplitude": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1}),
#             }
#         }

#     RETURN_TYPES = ("MASK",)
#     FUNCTION = "main_function"
#     CATEGORY = "/RyanOnTheInside/masks/"

#     def __init__(self):
#         super().__init__()
#         self.current_particle_size = 0
#         self.current_emission_rate = 0

#     def process_single_mask(self, mask: np.ndarray, frame_index: int, **kwargs) -> np.ndarray:
#         self.modulate_parameters(frame_index, **kwargs)
        
#         particle_mask = np.zeros_like(mask, dtype=np.float32)
#         for particle in self.particles:
#             particle_mask = self.draw_particle(particle_mask, particle)

#         result_mask = np.maximum(mask, particle_mask)
#         return result_mask

#     def modulate_parameters(self, frame_index, **kwargs):
#         t = frame_index / 30.0  # Assuming 30 fps

#         # Modulate particle size
#         base_size = float(kwargs['particle_size'])
#         size_freq = float(kwargs['size_frequency'])
#         size_amp = float(kwargs['size_amplitude'])
#         self.current_particle_size = base_size + size_amp * math.sin(2 * math.pi * size_freq * t)

#         # Modulate emission rate
#         base_rate = float(kwargs['emission_rate'])
#         rate_freq = float(kwargs['emission_frequency'])
#         rate_amp = float(kwargs['emission_amplitude'])
#         self.current_emission_rate = max(0, base_rate + rate_amp * math.sin(2 * math.pi * rate_freq * t))

#     def emit_particle(self):
#         # Use the current modulated size
#         angle = random.uniform(self.particle_direction - self.particle_spread/2, 
#                                self.particle_direction + self.particle_spread/2)
#         velocity = pymunk.Vec2d(math.cos(angle), math.sin(angle)) * self.particle_speed
        
#         mass = 1
#         radius = self.current_particle_size / 2
#         moment = pymunk.moment_for_circle(mass, 0, radius)
#         particle = pymunk.Body(mass, moment)
#         particle.position = self.emitter_pos
#         particle.velocity = velocity
#         particle.creation_time = self.space.current_time_step
#         particle.lifetime = self.particle_lifetime
#         particle.size = self.current_particle_size
        
#         shape = pymunk.Circle(particle, radius)
#         shape.elasticity = 0.9
#         shape.friction = 0.5
        
#         self.space.add(particle, shape)
#         self.particles.append(particle)
#         self.total_particles_emitted += 1

#     def update_particle_system(self, dt: float, current_mask: np.ndarray, respect_mask_boundary: bool):
#         if respect_mask_boundary:
#             self.update_mask_boundary(current_mask)
        
#         sub_steps = 5
#         sub_dt = dt / sub_steps
        
#         for _ in range(sub_steps):
#             # Use the current modulated emission rate
#             self.particles_to_emit += self.current_emission_rate * sub_dt
#             while self.particles_to_emit >= 1 and self.total_particles_emitted < self.max_particles:
#                 self.emit_particle()
#                 self.particles_to_emit -= 1
            
#             for particle in self.particles:
#                 old_pos = particle.position
#                 new_pos = old_pos + particle.velocity * sub_dt
#                 if respect_mask_boundary:
#                     self.check_particle_mask_collision(particle, old_pos, new_pos)
#                 particle.position = new_pos
            
#             self.space.step(sub_dt)
        
#         current_time = self.space.current_time_step
#         self.particles = [p for p in self.particles if current_time - p.creation_time < p.lifetime]

#     def main_function(self, masks, strength, invert, subtract_original, grow_with_blur, **kwargs):
#         self.particles = []  # Initialize particles to an empty list
#         self.modulate_parameters(0, **kwargs)  # Initialize modulated parameters
#         return super().main_function(masks, strength, invert, subtract_original, grow_with_blur, **kwargs)
