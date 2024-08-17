import torch
import numpy as np
from tqdm import tqdm
from comfy.utils import ProgressBar
from .mask_utils import (
    create_distance_transform, 
    normalize_array, 
    apply_blur, 
    apply_easing, 
    calculate_optical_flow, 
    apply_blur, 
    normalize_array
    )
from abc import ABC, abstractmethod
import pymunk 
import math
import random
from typing import List, Tuple
import pymunk
import cv2


class MaskBase(ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "subtract_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grow_with_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    CATEGORY = "/RyanOnTheInside/masks/"

    def __init__(self):
        self.pre_processors = []
        self.post_processors = []
        self.progress_bar = None
        self.tqdm_bar = None
        self.current_progress = 0
        self.total_steps = 0

    def add_pre_processor(self, func):
        self.pre_processors.append(func)
        return self

    def add_post_processor(self, func):
        self.post_processors.append(func)
        return self

    def pre_process(self, mask):
        for processor in self.pre_processors:
            mask = processor(mask)
        return mask

    def post_process(self, mask):
        for processor in self.post_processors:
            mask = processor(mask)
        return mask

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)
        self.tqdm_bar = tqdm(total=total_steps, desc=desc, leave=False)
        self.current_progress = 0
        self.total_steps = total_steps

    def update_progress(self, step=1):
        self.current_progress += step
        if self.progress_bar:
            self.progress_bar.update(step)
        if self.tqdm_bar:
            self.tqdm_bar.update(step)

    def end_progress(self):
        if self.tqdm_bar:
            self.tqdm_bar.close()
        self.progress_bar = None
        self.tqdm_bar = None
        self.current_progress = 0
        self.total_steps = 0

    @abstractmethod
    def process_mask(self, mask: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        """
        Process a single mask. This method must be implemented by child classes.
        """
        pass

    def apply_mask_operation(self, processed_masks: torch.Tensor, original_masks: torch.Tensor, strength: float, invert: bool, subtract_original: float, grow_with_blur: float, **kwargs) -> Tuple[torch.Tensor]:
        processed_masks_np = processed_masks.cpu().numpy() if isinstance(processed_masks, torch.Tensor) else processed_masks
        original_masks_np = original_masks.cpu().numpy() if isinstance(original_masks, torch.Tensor) else original_masks
        num_frames = processed_masks_np.shape[0]

        self.start_progress(num_frames, desc="Applying mask operation")

        result = []
        for processed_mask, original_mask in zip(processed_masks_np, original_masks_np):
            # Pre-processing
            processed_mask = self.pre_process(processed_mask)

            if invert:
                processed_mask = 1 - processed_mask

            if grow_with_blur > 0:
                processed_mask = apply_blur(processed_mask, grow_with_blur)

            # Apply subtract_original as the final step
            if subtract_original > 0:
                dist_transform = create_distance_transform(original_mask)
                dist_transform = normalize_array(dist_transform)
                threshold = 1 - subtract_original
                subtraction_mask = dist_transform > threshold
                processed_mask[subtraction_mask] = 0

            # Post-processing
            processed_mask = self.post_process(processed_mask)

            # Ensure the final mask is clipped between 0 and 1
            processed_mask = np.clip(processed_mask, 0, 1)

            result.append(processed_mask)
            self.update_progress()

        self.end_progress()

        return torch.from_numpy(np.stack(result)).float()

    @abstractmethod
    def main_function(self, *args, **kwargs) -> Tuple[torch.Tensor]:
        """
        Main entry point for the node. This method must be implemented by child classes.
        It should call apply_mask_operation with the appropriate arguments.
        """
        pass

class TemporalMaskBase(MaskBase, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()
        return {
            "required": {
                **parent_inputs["required"],
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "effect_duration": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "temporal_easing": ([ "ease_in_out","linear", "bounce", "elastic", "none"],),
                "palindrome": ("BOOLEAN", {"default": False}),
            }
        }

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process_single_mask(self, mask: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        """
        Process a single mask frame. This method must be implemented by child classes.
        frame_index is available in kwargs if needed.
        """
        pass

    def process_mask(self, mask: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        return self.process_single_mask(mask, strength, **kwargs)

    def apply_temporal_mask_operation(self, masks: torch.Tensor, strength: float, start_frame: int, end_frame: int, effect_duration: int, temporal_easing: str, palindrome: bool, **kwargs) -> Tuple[torch.Tensor]:
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        num_frames = masks_np.shape[0]
        
        end_frame = end_frame if end_frame > 0 else num_frames
        effect_duration = min(effect_duration, num_frames) if effect_duration > 0 else (end_frame - start_frame)
        if temporal_easing == "None":
            easing_values = np.ones(effect_duration)
        elif palindrome:
            half_duration = effect_duration // 2
            t = np.linspace(0, 1, half_duration)
            easing_values = apply_easing(t, temporal_easing)
            easing_values = np.concatenate([easing_values, easing_values[::-1]])
        else:
            t = np.linspace(0, 1, effect_duration)
            easing_values = apply_easing(t, temporal_easing)
        
        self.start_progress(num_frames, desc="Applying temporal mask operation")
        
        result = []
        for i in range(num_frames):
            if i < start_frame or i >= end_frame:
                result.append(masks_np[i])
            else:
                frame_in_effect = i - start_frame
                progress = easing_values[frame_in_effect % len(easing_values)]
                temporal_strength = strength * progress
                processed_mask = self.process_single_mask(masks_np[i], temporal_strength, frame_index=i, **kwargs)
                result.append(processed_mask)
            
            self.update_progress()
        
        self.end_progress()
        
        return (torch.from_numpy(np.stack(result)).float(),)

    def main_function(self, masks, strength, invert, subtract_original, grow_with_blur, start_frame, end_frame, effect_duration, temporal_easing, palindrome, **kwargs):
        original_masks = masks
        processed_masks = self.apply_temporal_mask_operation(masks, strength, start_frame, end_frame, effect_duration, temporal_easing, palindrome, **kwargs)
        ret = (self.apply_mask_operation(processed_masks[0], original_masks, strength, invert, subtract_original, grow_with_blur, **kwargs),)
        return ret


#TODO clean up the hamfisted resetting of all attributes
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
    CATEGORY = "/RyanOnTheInside/masks/"

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
        
        self.setup_particle_system(width, height, **kwargs)
        
        respect_mask_boundary = kwargs.get('respect_mask_boundary', False)
        
        self.start_progress(num_frames, desc="Processing particle system mask")
        
        mask_result = []
        image_result = []
        for i in range(num_frames):
            if i < start_frame or i >= end_frame:
                mask_result.append(masks_np[i])
                image_result.append(np.stack([masks_np[i]] * 3, axis=-1))
            else:
                self.update_particle_system(1.0 / 30.0, masks_np[i], respect_mask_boundary)
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
            initial_plume = float(emitter['initial_plume'])
            initial_particle_count = int(self.max_particles * initial_plume / len(self.emitters))
            
            # Create initial plume of particles for this emitter
            for _ in range(initial_particle_count):
                self.emit_particle(emitter, height, width, emitter_index)

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

        self.setup_spring_joints()
        

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

    def update_particle_system(self, dt: float, current_mask: np.ndarray, respect_mask_boundary: bool):
        self.total_time += dt
        if respect_mask_boundary:
            self.update_mask_boundary(current_mask)
        
        height, width = current_mask.shape
        self.update_vortices(width, height, dt)
        sub_steps = 5
        sub_dt = dt / sub_steps
        
        for _ in range(sub_steps):
            for i, emitter in enumerate(self.emitters):
                emission_rate = emitter['emission_rate'] * 10  # Increased sensitivity
                self.particles_to_emit[i] += emission_rate * sub_dt
                while self.particles_to_emit[i] >= 1 and self.total_particles_emitted < self.max_particles:
                    self.emit_particle(emitter, height, width, i)
                    self.particles_to_emit[i] -= 1
            
            particles_before = len(self.particles)
            self.particles = [p for p in self.particles if self.total_time - p.creation_time < p.lifetime]
            particles_after = len(self.particles)
            particles_removed = particles_before - particles_after
 
            
            for particle in self.particles:
                self.apply_vortex_force(particle, sub_dt)
                self.apply_gravity_well_force(particle)
                old_pos = particle.position
                new_pos = old_pos + particle.velocity * sub_dt
                if respect_mask_boundary:
                    self.check_particle_mask_collision(particle, old_pos, new_pos)
                particle.position = new_pos
            

            self.space.step(sub_dt)
        self.setup_spring_joints()

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

    def emit_particle(self, emitter, height, width, emitter_index):
        emitter_pos = (float(emitter['emitter_x']) * width, float(emitter['emitter_y']) * height)
        particle_direction = math.radians(float(emitter['particle_direction']))
        particle_spread = math.radians(float(emitter['particle_spread']))
        particle_speed = float(emitter['particle_speed'])
        particle_size = float(emitter['particle_size'])

        angle = random.uniform(particle_direction - particle_spread/2, 
                               particle_direction + particle_spread/2)
        velocity = pymunk.Vec2d(math.cos(angle), math.sin(angle)) * particle_speed
        
        mass = 1
        radius = particle_size / 2
        moment = pymunk.moment_for_circle(mass, 0, radius)
        particle = pymunk.Body(mass, moment)
        particle.position = emitter_pos
        particle.velocity = velocity
        particle.creation_time = self.total_time
        particle.lifetime = self.particle_lifetime
        particle.size = particle_size
        particle.color = emitter['color']
        particle.emitter_index = emitter_index
        
        shape = pymunk.Circle(particle, radius)
        shape.elasticity = 0.9
        shape.friction = 0.5
        
        self.space.add(particle, shape)
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
        radius = int(particle.size / 2)
        cv2.circle(mask, (x, y), radius, 1, -1)
        cv2.circle(image, (x, y), radius, particle.color, -1)
        return mask, image

    def modulate_parameters(self, frame_index, mask):
        t = frame_index / 30.0  # 30 fps #TODO fix fps hack and add particle specific modulation
        height, width = mask.shape

        for emitter in self.emitters:
            if 'movement' in emitter:
                movement = emitter['movement']
                
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
        for emitter in emitters:
            emitter['initial_x'] = emitter['emitter_x']
            emitter['initial_y'] = emitter['emitter_y']
            if 'movement' in emitter:
                emitter['initial_direction'] = emitter['particle_direction']
            self.emitters.append(emitter)
        
        original_masks = masks
        processed_masks, processed_images = self.process_mask(masks, emitters=self.emitters, **kwargs)
        result_masks = self.apply_mask_operation(processed_masks, original_masks, strength, invert, subtract_original, grow_with_blur, **kwargs)
        return (result_masks, processed_images,)
    

class OpticalFlowMaskBase(MaskBase, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "images": ("IMAGE",),
                "flow_method": (["Farneback", "LucasKanade", "PyramidalLK"],),
                "flow_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "magnitude_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    CATEGORY = "/RyanOnTheInside/masks/"

    def __init__(self):
        super().__init__()

    def process_mask(self, mask: np.ndarray, strength: float, images: np.ndarray, flow_method: str, flow_threshold: float, magnitude_threshold: float, frame_index: int, **kwargs) -> np.ndarray:
        if frame_index == 0 or frame_index >= len(images) - 1:
            return mask

        frame1 = (images[frame_index] * 255).astype(np.uint8)
        frame2 = (images[frame_index + 1] * 255).astype(np.uint8)
        
        flow = calculate_optical_flow(frame1, frame2, flow_method)
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        flow_magnitude[flow_magnitude < flow_threshold] = 0
        
        flow_magnitude[flow_magnitude < magnitude_threshold * np.max(flow_magnitude)] = 0
        
        flow_magnitude = normalize_array(flow_magnitude)

        return self.apply_flow_mask(mask, flow_magnitude, flow, strength, **kwargs)

    @abstractmethod
    def apply_flow_mask(self, mask: np.ndarray, flow_magnitude: np.ndarray, flow: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        """
        Apply the optical flow-based mask operation. To be implemented by subclasses.
        """
        pass

    def main_function(self, masks, images, strength, flow_method, flow_threshold, magnitude_threshold, **kwargs):
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        images_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
        
        num_frames = masks_np.shape[0]
        self.start_progress(num_frames, desc="Applying optical flow mask operation")

        result = []
        for i in range(num_frames):
            processed_mask = self.process_mask(masks_np[i], strength, images_np, flow_method, flow_threshold, magnitude_threshold, frame_index=i, **kwargs)
            result.append(processed_mask)
            self.update_progress()

        self.end_progress()

        processed_masks = np.stack(result)
        return self.apply_mask_operation(processed_masks, masks, strength, **kwargs)



