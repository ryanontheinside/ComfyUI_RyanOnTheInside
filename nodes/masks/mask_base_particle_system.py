import torch
import numpy as np
from .mask_utils import (
    create_distance_transform, 
    normalize_array, 
    apply_easing, 
    normalize_array
    )
from abc import ABC, abstractmethod
import pymunk 
import math
import random
from typing import List, Tuple
import pymunk
import cv2
from .mask_base import MaskBase
from ...tooltips import apply_tooltips


#TODO clean up the hamfisted resetting of all attributes
@apply_tooltips
class ParticleSystemMaskBase(MaskBase, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()
        return {
            "required": {
                **parent_inputs["required"],
                "emitters": ("PARTICLE_EMITTER",),
                "particle_count": ("INT", {"default": 200, "min": 1, "max": 10000, "step": 1}),
                "particle_lifetime": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "wind_strength": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "wind_direction": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "gravity": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 1.0}),
                "warmup_period": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "respect_mask_boundary": ("BOOLEAN", {"default": False}),
            },
            "optional":{
                "vortices": ("VORTEX",),
                "wells": ("GRAVITY_WELL",),
                "static_bodies": ("STATIC_BODY",),
                "well_strength_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    FUNCTION = "main_function"
    CATEGORY="RyanOnTheInside/ParticleSystems"

    def __init__(self):
        super().__init__()
        self.initialize()
        self.total_time = 0  
        

    def initialize(self):
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0, 0)
        self.particles: List[pymunk.Body] = []
        self.mask_shapes: List[pymunk.Shape] = []
        self.particles_to_emit = [] 
        self.total_particles_emitted = 0
        self.max_particles = 0
        self.emitters = []
        self.gravity_wells = []
        self.spring_joints = []
        self.static_bodies = []
        self.vortices = []
        self.total_time = 0

    @abstractmethod
    def process_single_mask(self, mask: np.ndarray, frame_index: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single mask frame. This method must be implemented by child classes.
        It should return both the processed mask and the corresponding color image.
        """
        pass

    def process_mask(self, masks: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        num_frames, height, width = masks_np.shape
        
        start_frame = kwargs.get('start_frame', 0)
        end_frame = kwargs.get('end_frame', num_frames)
        end_frame = end_frame if end_frame > 0 else num_frames
        warmup_period = kwargs.get('warmup_period', 0)
        
        self.setup_particle_system(width, height, **kwargs)
        
        respect_mask_boundary = kwargs.get('respect_mask_boundary', False)
        
        self.start_progress(num_frames + warmup_period, desc="Processing particle system mask")
        
        # Run warmup period
        for _ in range(warmup_period):
            self.update_particle_system(1.0 / 30.0, masks_np[0], respect_mask_boundary, -1)
            self.update_progress()
        
        mask_result = []
        image_result = []
        for i in range(num_frames):
            if i < start_frame or i >= end_frame:
                mask_result.append(masks_np[i])
                image_result.append(np.stack([masks_np[i]] * 3, axis=-1))
            else:
                self.update_particle_system(1.0 / 30.0, masks_np[i], respect_mask_boundary, i)
                processed_mask, processed_image = self.process_single_mask(masks_np[i], frame_index=i, **kwargs)
                processed_image = self.draw_static_bodies(processed_image)
                processed_mask, processed_image = self.draw_vortices(processed_mask, processed_image)
                processed_mask, processed_image = self.draw_gravity_wells(processed_mask, processed_image)
                mask_result.append(processed_mask)
                image_result.append(processed_image)
            
            self.update_progress()
        
        self.end_progress()
        
        return torch.from_numpy(np.stack(mask_result)).float(), torch.from_numpy(np.stack(image_result)).float()


    def setup_particle_system(self, width: int, height: int, **kwargs):
        self.space.gravity = pymunk.Vec2d(float(kwargs['wind_strength']), float(kwargs['gravity']))
        
        # Set up spatial hash for efficient collision detection
        cell_size = max(emitter['particle_size'] for emitter in kwargs['emitters']) * 2.5
        self.space.use_spatial_hash(cell_size, int((width * height) / (cell_size * cell_size)))
        
        self.emitters = kwargs['emitters']
        self.max_particles = int(kwargs['particle_count'])
        self.particle_lifetime = float(kwargs['particle_lifetime'])
        
        self.particles_to_emit = [0] * len(self.emitters)
        self.total_particles_emitted = 0

        for emitter_index, emitter in enumerate(self.emitters):
            emitter_pos = (float(emitter['emitter_x']) * width, float(emitter['emitter_y']) * height)
            emitter['color'] = self.string_to_rgb(emitter['color'])
            emitter['original_emission_rate'] = emitter['emission_rate']  # Store original emission rate
            initial_plume = float(emitter['initial_plume'])
            initial_particle_count = int(self.max_particles * initial_plume / len(self.emitters))
            
            # Create initial plume of particles for this emitter
            for _ in range(initial_particle_count):
                self.emit_particle(emitter, height, width, emitter_index, 0)

        # Initialize emitter modulations
        self.prepare_emitter_modulations()

        # Check for provided wells and vortices
        if 'vortices' in kwargs:
            self.initialize_vortices(width, height, kwargs['vortices'])
        else:
            self.vortices = []  
        if 'wells' in kwargs:
            self.initialize_gravity_wells(width, height, kwargs['wells'])
        else:
            self.gravity_wells = []
        if 'static_bodies' in kwargs:
            static_bodies = kwargs['static_bodies'] or []
            self.initialize_static_bodies(width, height, static_bodies)
        else:
            self.static_bodies = []
        
        self.well_strength_multiplier = kwargs.get('well_strength_multiplier', 1.0)
        self.prepare_emitter_particle_modulations()
        self.setup_spring_joints()
        

    def prepare_emitter_modulations(self):
        for emitter in self.emitters:
            emitter_modulation_chain = emitter.get("emitter_modulation", [])
            if emitter_modulation_chain:
                prepared_modulations = []
                for modulation in emitter_modulation_chain:
                    modulation_type = modulation["type"]
                    if modulation_type == "EmitterEmissionRateModulation":
                        prepared_modulations.append({
                            "type": "emission_rate",
                            "start_frame": modulation["start_frame"],
                            "end_frame": modulation["end_frame"],
                            "effect_duration": modulation["effect_duration"],
                            "temporal_easing": modulation["temporal_easing"],
                            "palindrome": modulation["palindrome"],
                            "random": modulation["random"],
                            "target_emission_rate": modulation["target_emission_rate"],
                            "feature": modulation.get("feature"),
                        })
                    # Add more modulation types as needed
                emitter["emitter_modulations"] = prepared_modulations
            else:
                emitter['emitter_modulations'] = []


    def ease_in_out(self, t):
        # Simple ease in/out function
        return t * t * (3 - 2 * t)

    @staticmethod
    def string_to_rgb(color_string):
        if isinstance(color_string, tuple):
            return color_string
        color_values = color_string.strip('()').split(',')
        return tuple(int(value.strip()) / 255.0 for value in color_values)

###START FORCES AND SHIT
    def initialize_vortices(self, width, height, vortices):
        self.vortices = []
        for vortex in vortices:
            x = vortex['x'] * width
            y = vortex['y'] * height
            vortex_obj = {
                'position': pymunk.Vec2d(x, y),
                'velocity': pymunk.Vec2d(random.uniform(-1, 1), random.uniform(-1, 1)).normalized() * vortex['movement_speed'],
                'strength': vortex['strength'],
                'radius': vortex['radius'],
                'inward_factor': vortex['inward_factor'],
                'draw': vortex['draw'],
                'color': vortex['color'],
            }
            self.vortices.append(vortex_obj)

    def update_vortices(self, width, height, dt):
        for vortex in self.vortices:
            vortex['position'] += vortex['velocity'] * dt
            # Bounce off boundaries
            if vortex['position'].x < 0 or vortex['position'].x > width:
                vortex['velocity'].x *= -1
            if vortex['position'].y < 0 or vortex['position'].y > height:
                vortex['velocity'].y *= -1

    def apply_vortex_force(self, particle, dt):
        for vortex in self.vortices:
            offset = particle.position - vortex['position']
            distance = offset.length

            if distance < vortex['radius']:
                tangent = pymunk.Vec2d(-offset.y, offset.x).normalized()
                radial = -offset.normalized()
                strength = vortex['strength']
                
                tangential_force = tangent * strength * (distance / vortex['radius'])
                radial_force = radial * strength * vortex['inward_factor']
                
                total_force = tangential_force + radial_force
                
                particle.apply_force_at_local_point(total_force)

    def setup_spring_joints(self):
        for emitter_index, emitter in enumerate(self.emitters):
            if "spring_joint_setting" in emitter:
                setting = emitter["spring_joint_setting"]
                particles = [p for p in self.particles if p.emitter_index == emitter_index]
                
                for i, particle1 in enumerate(particles):
                    for particle2 in particles[i+1:]:
                        distance = (particle1.position - particle2.position).length
                        if distance <= setting["max_distance"]:
                            spring = pymunk.DampedSpring(
                                particle1, particle2, (0, 0), (0, 0), 
                                rest_length=setting["rest_length"], 
                                stiffness=setting["stiffness"], 
                                damping=setting["damping"]
                            )
                            self.space.add(spring)
                            self.spring_joints.append(spring)
    
    def initialize_gravity_wells(self, width, height, wells):
        self.gravity_wells = []
        for well in wells:
            x = well['x'] * width
            y = well['y'] * height
            well_obj = {
                'position': pymunk.Vec2d(x, y),
                'strength': well['strength'],
                'radius': well['radius'],
                'type': well['type'],  # 'attract' or 'repel'
                'draw':well['draw'],
                'color':well['color'],
            }
            self.gravity_wells.append(well_obj)

    def apply_gravity_well_force(self, particle):
        for well in self.gravity_wells:
            offset = well['position'] - particle.position
            distance = offset.length
            if distance < well['radius']:
                force_magnitude = well['strength'] * (1 - distance / well['radius']) * self.well_strength_multiplier
                force_direction = offset.normalized()
                if well['type'] == 'repel':
                    force_direction = -force_direction
                force = force_direction * force_magnitude
                particle.apply_force_at_world_point(force, particle.position)

    def initialize_static_bodies(self, width, height, static_bodies):
        pass
        for body in static_bodies:
            shape_type = body['shape_type']
            if shape_type == 'segment':
                p1 = (body['x1'] * width, body['y1'] * height)
                p2 = (body['x2'] * width, body['y2'] * height)
                shape = pymunk.Segment(self.space.static_body, p1, p2, 1)
            elif shape_type == 'circle':
                position = (body['x1'] * width, body['y1'] * height)
                shape = pymunk.Circle(self.space.static_body, body['radius'] * min(width, height), position)
            elif shape_type == 'polygon':
                points = [
                    (body['x1'] * width, body['y1'] * height),
                    (body['x2'] * width, body['y1'] * height),
                    (body['x2'] * width, body['y2'] * height),
                    (body['x1'] * width, body['y2'] * height)
                ]
                shape = pymunk.Poly(self.space.static_body, points)
            
            shape.elasticity = body['elasticity']
            shape.friction = body['friction']
            self.space.add(shape)
            self.static_bodies.append({
                'shape': shape,
                'draw': body['draw'],
                'color': self.string_to_rgb(body['color'])
            })

    def draw_static_bodies(self, image: np.ndarray) -> np.ndarray:
        for body in self.static_bodies:
            if body['draw']:
                shape = body['shape']
                color = body['color']
                if isinstance(shape, pymunk.Segment):
                    cv2.line(image, 
                            (int(shape.a.x), int(shape.a.y)), 
                            (int(shape.b.x), int(shape.b.y)), 
                            color, 2)
                elif isinstance(shape, pymunk.Circle):
                    cv2.circle(image, 
                            (int(shape.body.position.x), int(shape.body.position.y)),                            int(shape.radius), 
                            color, -1)
                elif isinstance(shape, pymunk.Poly):
                    points = np.array([(int(v.x), int(v.y)) for v in shape.get_vertices()], np.int32)
                    cv2.fillPoly(image, [points], color)
        return image

    def draw_vortices(self, mask: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for vortex in self.vortices:
            if vortex['draw'] > 0:
                thickness = max(1, int(vortex['draw'] * 5))
                color = self.string_to_rgb(vortex['color'])
                center = (int(vortex['position'].x), int(vortex['position'].y))
                radius = int(vortex['radius'])
                cv2.circle(mask, center, radius, 1.0, thickness)
                cv2.circle(image, center, radius, color, thickness)
        return mask, image

    def draw_gravity_wells(self, mask: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for well in self.gravity_wells:
            if well['draw'] > 0:
                thickness = max(1, int(well['draw'] * 5))
                color = self.string_to_rgb(well['color'])
                center = (int(well['position'].x), int(well['position'].y))
                radius = int(well['radius'])
                cv2.circle(mask, center, radius, 1.0, thickness)
                cv2.circle(image, center, radius, color, thickness)
        return mask, image
    
###END FORCES AND SHIT

###START PARTICLE MODULATION

    def calculate_modulation_frames(self, start_frame, end_frame, effect_duration):
        start_frame = max(0, start_frame)
        
        if end_frame <= 0 or end_frame > self.total_frames:
            end_frame = self.total_frames
        
        if end_frame <= start_frame:
            return None, None, None  # Invalid frame range
        
        if effect_duration <= 0:
            effect_duration = end_frame - start_frame
        else:
            effect_duration = min(effect_duration, end_frame - start_frame)
        
        return start_frame, end_frame, effect_duration

    def prepare_emitter_particle_modulations(self):
        self.emitter_modulations = {}
        for emitter_index, emitter in enumerate(self.emitters):
            if 'particle_modulation' in emitter and emitter['particle_modulation']:
                processed_modulations = []
                for modulation in emitter['particle_modulation']:
                    start_frame, end_frame, effect_duration = self.calculate_modulation_frames(
                        modulation['start_frame'],
                        modulation['end_frame'],
                        modulation['effect_duration']
                    )
                    
                    if start_frame is None:
                        continue  # Skip this modulation as it's invalid
                    
                    processed_mod = {
                        'type': modulation['type'],
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'effect_duration': effect_duration,
                        'temporal_easing': modulation['temporal_easing'],
                        'palindrome': modulation['palindrome'],
                        'random': modulation.get('random', False),
                        'feature': modulation.get('feature', None)
                    }
                    
                    if modulation['type'] == 'ParticleSizeModulation':
                        processed_mod['target_size'] = modulation['target_size']
                    elif modulation['type'] == 'ParticleSpeedModulation':
                        processed_mod['target_speed'] = modulation['target_speed']
                    elif modulation['type'] == 'ParticleColorModulation':
                        processed_mod['target_color'] = self.string_to_rgb(modulation['target_color'])
                    
                    processed_modulations.append(processed_mod)
                
                if processed_modulations:  # Only add if there are valid modulations
                    self.emitter_modulations[emitter_index] = processed_modulations

    def apply_particle_modulations(self, particle, current_frame):
        emitter_index = particle.emitter_index
        if emitter_index not in self.emitter_modulations:
            return

        for modulation in self.emitter_modulations[emitter_index]:
            if modulation['start_frame'] <= current_frame < modulation['end_frame']:
                if modulation.get('random', False):
                    progress = random.random()  # Generate a random value between 0 and 1
                elif 'feature' in modulation and modulation['feature'] is not None:
                    progress = modulation['feature'].get_value_at_frame(current_frame)
                else:
                    progress = self.calculate_modulation_progress(current_frame, modulation, particle.creation_frame)
                
                if modulation['type'] == 'ParticleSizeModulation':
                    self.apply_size_modulation(particle, modulation, progress)
                elif modulation['type'] == 'ParticleSpeedModulation':
                    self.apply_speed_modulation(particle, modulation, progress)
                elif modulation['type'] == 'ParticleColorModulation':
                    self.apply_color_modulation(particle, modulation, progress)

    def calculate_modulation_progress(self, current_frame, modulation, particle_creation_frame):
        start_frame = modulation['start_frame']
        duration = modulation['effect_duration']
        
        progress = (max(0, current_frame) - start_frame) / duration
        
        if modulation['palindrome']:
            if progress <= 0.5:
                progress = progress * 2
            else:
                progress = (1 - progress) * 2
        
        # cllamp progress between 0 and 1
        progress = max(0, min(1, progress))
        
        eased_progress = apply_easing(progress, modulation['temporal_easing'])
        return eased_progress

    def apply_size_modulation(self, particle, modulation, progress):
        emitter = self.emitters[particle.emitter_index]
        original_size = emitter['particle_size']
        target_size = modulation['target_size']
        new_size = original_size + (target_size - original_size) * progress
        
        particle.size = new_size
        
        # Update the Pymunk Shape so they fdont overlap when resized
        new_radius = new_size / 2
        self.space.remove(particle.shape)
        particle.shape = pymunk.Circle(particle, new_radius)
        self.space.add(particle.shape)

    def apply_speed_modulation(self, particle, modulation, progress):
        emitter = self.emitters[particle.emitter_index]
        original_speed = emitter['particle_speed']
        target_speed = modulation['target_speed']
        new_speed = original_speed + (target_speed - original_speed) * progress
        if particle.velocity.length > 0:
            particle.velocity = particle.velocity.normalized() * new_speed

    def apply_color_modulation(self, particle, modulation, progress):
        emitter = self.emitters[particle.emitter_index]
        original_color=emitter['color']
        target_color = modulation['target_color']
        particle.color = tuple(o + (t - o) * progress for o, t in zip(original_color, target_color))

###END PARTICLE MODULATION

    def update_particle_system(self, dt: float, current_mask: np.ndarray, respect_mask_boundary: bool, frame_index: int):
        self.total_time += dt
        if respect_mask_boundary:
            self.update_mask_boundary(current_mask)
        
        height, width = current_mask.shape
        self.update_vortices(width, height, dt)
        sub_steps = 5
        sub_dt = dt / sub_steps
        
        for _ in range(sub_steps):
            for i, emitter in enumerate(self.emitters):
                # Apply emitter modulations
                self.apply_emitter_modulations(emitter, frame_index)
                
                # Check if the emitter is active in the current frame
                if emitter['start_frame'] <= max(0, frame_index) and (emitter['end_frame'] == 0 or max(0, frame_index) < emitter['end_frame']):
                    emission_rate = emitter['emission_rate'] * 10  # Increased sensitivity
                    self.particles_to_emit[i] += emission_rate * sub_dt
                    while self.particles_to_emit[i] >= 1 and self.total_particles_emitted < self.max_particles:
                        self.emit_particle(emitter, height, width, i, frame_index)
                        self.particles_to_emit[i] -= 1
            
            particles_before = len(self.particles)
            self.particles = [p for p in self.particles if max(0, frame_index) - p.creation_frame < self.particle_lifetime * 30]  # Assuming 30 fps
            particles_after = len(self.particles)
            particles_removed = particles_before - particles_after
            
            for particle in self.particles:
                self.apply_vortex_force(particle, sub_dt)
                self.apply_gravity_well_force(particle)
                
                # Apply modulations before updating position
                self.apply_particle_modulations(particle, max(0, frame_index))
                
                old_pos = particle.position
                new_pos = old_pos + particle.velocity * sub_dt
                if respect_mask_boundary:
                    self.check_particle_mask_collision(particle, old_pos, new_pos)
                particle.position = new_pos
            
            self.space.step(sub_dt)
        
        self.setup_spring_joints()

    def apply_emitter_modulations(self, emitter, frame_index):
        for modulation in emitter.get("emitter_modulations", []):
            start_frame, end_frame, effect_duration = self.calculate_modulation_frames(
                modulation["start_frame"],
                modulation["end_frame"],
                modulation["effect_duration"]
            )
            
            if start_frame is None:
                continue  # Skip this modulation as it's invalid
            
            if start_frame <= frame_index <= end_frame:
                original_emission_rate = emitter.get("original_emission_rate", emitter["emission_rate"])
                target_emission_rate = modulation["target_emission_rate"]

                if modulation.get('random', False):
                    factor = random.random()  # Generate a random value between 0 and 1
                elif 'feature' in modulation and modulation['feature'] is not None:
                    factor = modulation['feature'].get_value_at_frame(frame_index)
                else:
                    # Compute modulation progress
                    progress = (frame_index - start_frame) / effect_duration
                    if modulation["temporal_easing"] == "linear":
                        factor = progress
                    elif modulation["temporal_easing"] == "ease_in_out":
                        factor = self.ease_in_out(progress)
                    else:
                        factor = progress  # Default to linear

                # Normalize the factor between original and target emission rates
                emitter["emission_rate"] = original_emission_rate + (target_emission_rate - original_emission_rate) * factor
    
    def ease_in_out(self, t):
        # Simple ease in/out function
        return t * t * (3 - 2 * t)

    def check_particle_mask_collision(self, particle, old_pos, new_pos):
        for segment in self.mask_shapes:
            if self.line_segment_intersect(old_pos, new_pos, pymunk.Vec2d(*segment.a), pymunk.Vec2d(*segment.b)):
                segment_vec = pymunk.Vec2d(*segment.b) - pymunk.Vec2d(*segment.a)
                normal = segment_vec.perpendicular_normal()
                v = particle.velocity
                reflection = v - 2 * v.dot(normal) * normal
                
                particle.velocity = reflection * 0.9  # Reduce velocity slightly on bounce
                
                penetration_depth = (new_pos - old_pos).dot(normal)
                particle.position = old_pos + normal * (penetration_depth + particle.size / 2 + 1)
                break

    def line_segment_intersect(self, p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)
        
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

    def emit_particle(self, emitter, height, width, emitter_index, frame_index):
        emitter_x = float(emitter['emitter_x']) * width
        emitter_y = float(emitter['emitter_y']) * height
        particle_direction = math.radians(float(emitter['particle_direction']))
        particle_spread = math.radians(float(emitter['particle_spread']))
        particle_speed = float(emitter['particle_speed'])
        particle_size = float(emitter['particle_size'])
        
        emission_radius = float(emitter.get('emission_radius', 0))  # Default to 0 if not specified

        # Generate a random point within the circular emission area
        if emission_radius > 0:
            r = random.uniform(0, emission_radius)
            theta = random.uniform(0, 2 * math.pi)
            emitter_pos = (
                emitter_x + r * math.cos(theta),
                emitter_y + r * math.sin(theta)
            )
        else:
            emitter_pos = (emitter_x, emitter_y)

        angle = random.uniform(particle_direction - particle_spread/2, 
                               particle_direction + particle_spread/2)
        velocity = pymunk.Vec2d(math.cos(angle), math.sin(angle)) * particle_speed
        
        mass = 1
        radius = particle_size / 2
        moment = pymunk.moment_for_circle(mass, 0, radius)
        particle = pymunk.Body(mass, moment)
        particle.position = emitter_pos
        particle.velocity = velocity
        particle.lifetime = self.particle_lifetime
        particle.size = particle_size
        particle.color = emitter['color']
        particle.emitter_index = emitter_index
        particle.creation_frame = max(0, frame_index)  # Ensure non-negative creation frame
        
        shape = pymunk.Circle(particle, radius)
        shape.elasticity = 0.9
        shape.friction = 0.5
        
        self.space.add(particle, shape)
        particle.shape = shape  # Store the shape with the particle
        self.particles.append(particle)
        self.total_particles_emitted += 1

    def update_mask_boundary(self, mask: np.ndarray):

        #TODO get the segments contiguous

        for shape in self.mask_shapes:
            self.space.remove(shape)
        self.mask_shapes.clear()

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Simplify the contour
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
        
            points = [tuple(map(float, point[0])) for point in approx]
            
            if len(points) < 3:
                continue
            
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                segment = pymunk.Segment(self.space.static_body, p1, p2, 1)
                segment.elasticity = 0.9
                segment.friction = 0.5
                self.space.add(segment)
                self.mask_shapes.append(segment)

    def draw_particle(self, mask: np.ndarray, image: np.ndarray, particle: pymunk.Body) -> Tuple[np.ndarray, np.ndarray]:
        x, y = int(particle.position.x), int(particle.position.y)
        radius = int(particle.shape.radius)  # Use the shape's radius for drawing
        cv2.circle(mask, (x, y), radius, 1, -1)
        cv2.circle(image, (x, y), radius, particle.color, -1)
        return mask, image

    def modulate_parameters(self, frame_index, mask):
        t = frame_index / 30.0  # 30 fps #TODO fix fps hack and add particle specific modulation
        height, width = mask.shape

        for emitter in self.emitters:
            if 'movement' in emitter:
                movement = emitter['movement']
                
                # Check if a feature is present and apply it to the specified parameter
                if 'feature' in movement and movement['feature'] is not None:
                    feature = movement['feature']
                    feature_param = movement['feature_param']
                    feature_value = feature.get_value_at_frame(frame_index)
                    
                    if feature_param in ['emitter_x_frequency', 'emitter_y_frequency', 'direction_frequency']:
                        # Use 1 + feature_value to ensure the frequency is always at least the original value
                        movement[feature_param] *= (1 + feature_value)

                # Modulate emitter_x
                base_x = float(emitter['initial_x'])
                freq_x = float(movement['emitter_x_frequency'])
                amp_x = float(movement['emitter_x_amplitude'])
                emitter['emitter_x'] = base_x + amp_x * math.sin(2 * math.pi * freq_x * t)

                # Modulate emitter_y
                base_y = float(emitter['initial_y'])
                freq_y = float(movement['emitter_y_frequency'])
                amp_y = float(movement['emitter_y_amplitude'])
                emitter['emitter_y'] = base_y + amp_y * math.sin(2 * math.pi * freq_y * t)

                # Modulate particle_direction
                base_dir = float(emitter['initial_direction'])
                freq_dir = float(movement['direction_frequency'])
                amp_dir = float(movement['direction_amplitude'])
                emitter['particle_direction'] = base_dir + amp_dir * math.sin(2 * math.pi * freq_dir * t)

                # Update emitter position
                emitter['position'] = (emitter['emitter_x'] * width, emitter['emitter_y'] * height)

    def main_function(self, masks, strength, invert, subtract_original, grow_with_blur, emitters, **kwargs):
        self.initialize()
        self.total_frames = masks.shape[0] + kwargs.get('warmup_period', 0)  # Include warmup period in total frames
        for emitter in emitters:
            emitter['initial_x'] = emitter['emitter_x']
            emitter['initial_y'] = emitter['emitter_y']
            if 'movement' in emitter:
                emitter['initial_direction'] = emitter['particle_direction']
            self.emitters.append(emitter)
        
        original_masks = masks
        processed_masks, processed_images = self.process_mask(masks, emitters=self.emitters, **kwargs)
        
        if subtract_original > 0:
            original_masks_np = original_masks.cpu().numpy()
            for i in range(self.total_frames):
                mask_8bit = (original_masks_np[i] * 255).astype(np.uint8)
                dist_transform = create_distance_transform(mask_8bit)
                dist_transform = normalize_array(dist_transform)
                threshold = 1 - subtract_original
                subtraction_mask = dist_transform > threshold
                
                processed_images[i][subtraction_mask] = 0

        result_masks = self.apply_mask_operation(processed_masks, original_masks, strength, invert, subtract_original, grow_with_blur, **kwargs)
        
        return (result_masks, processed_images,)