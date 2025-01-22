import numpy as np
import cv2
import torch
from .mask_base import OpticalFlowMaskBase
from .mask_utils import calculate_optical_flow, apply_blur, normalize_array
from ...tooltips import apply_tooltips


#TODO make all this better.

@apply_tooltips
class OpticalFlowMaskModulation(OpticalFlowMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "modulation_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 20, "step": 1}),
                "trail_length": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "decay_factor": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "decay_style": (["fade", "thickness"],),
                "max_thickness": ("INT", {"default": 20, "min": 1, "max": 50, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_optical_flow_modulation"

    def __init__(self):
        super().__init__()
        self.trail_buffer = []

    def apply_flow_mask(self, mask: np.ndarray, flow_magnitude: np.ndarray, flow: np.ndarray, strength: float, modulation_strength: float, blur_radius: int, trail_length: int, decay_factor: float, decay_style: str, max_thickness: int, **kwargs) -> np.ndarray:
        if blur_radius > 0:
            flow_magnitude = apply_blur(flow_magnitude, blur_radius)

        # Initialize modulated_mask with the original mask
        modulated_mask = mask.copy()

        # Create smear effect
        smear = np.zeros_like(mask)
        for j, trail in enumerate(self.trail_buffer):
            decay = decay_factor ** (len(self.trail_buffer) - j)
            if decay_style == "fade":
                smear += trail * decay
            elif decay_style == "thickness":
                kernel_size = max(3, int(decay * max_thickness))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilated_trail = cv2.dilate(trail, kernel, iterations=1)
                smear = np.maximum(smear, dilated_trail)

        if decay_style == "fade":
            modulated_mask = np.maximum(modulated_mask, smear)
        elif decay_style == "thickness":
            modulated_mask = np.maximum(modulated_mask, smear > 0.5)  # Threshold for binary mask

        modulated_mask = np.clip(modulated_mask + modulation_strength * flow_magnitude * strength, 0, 1)

        # Ensure binary mask for thickness style
        if decay_style == "thickness":
            modulated_mask = (modulated_mask > 0.5).astype(np.float32)

        # Update trail buffer
        self.trail_buffer.append(flow_magnitude * mask)
        if len(self.trail_buffer) > trail_length:
            self.trail_buffer.pop(0)

        return modulated_mask

    def apply_optical_flow_modulation(self, masks, images, strength, flow_method, flow_threshold, magnitude_threshold, modulation_strength, blur_radius, trail_length, decay_factor, decay_style, max_thickness, **kwargs):
        self.trail_buffer = []  # Reset trail buffer
        return (super().main_function(masks, images, strength, flow_method, flow_threshold, magnitude_threshold, modulation_strength=modulation_strength, blur_radius=blur_radius, trail_length=trail_length, decay_factor=decay_factor, decay_style=decay_style, max_thickness=max_thickness, **kwargs),)
   
@apply_tooltips
class OpticalFlowDirectionMask(OpticalFlowMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "direction": (["horizontal", "vertical", "radial_in", "radial_out", "clockwise", "counterclockwise"],),
                "angle_threshold": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 180.0, "step": 1.0}),
                "blur_radius": ("INT", {"default": 3, "min": 0, "max": 20, "step": 1}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_direction_mask"

    def apply_flow_mask(self, mask: np.ndarray, flow_magnitude: np.ndarray, flow: np.ndarray, strength: float, direction: str, angle_threshold: float, blur_radius: int, invert: bool, **kwargs) -> np.ndarray:
        height, width = flow.shape[:2]
        y, x = np.mgrid[0:height, 0:width]

        angle = np.arctan2(flow[..., 1], flow[..., 0]) * 180 / np.pi
        angle[angle < 0] += 360  # Convert to 0-360 range

        if direction == "horizontal":
            mask = (np.abs(angle - 0) < angle_threshold) | (np.abs(angle - 180) < angle_threshold)
        elif direction == "vertical":
            mask = (np.abs(angle - 90) < angle_threshold) | (np.abs(angle - 270) < angle_threshold)
        elif direction == "radial_in":
            center_y, center_x = height // 2, width // 2
            radial_angle = np.arctan2(y - center_y, x - center_x) * 180 / np.pi
            radial_angle[radial_angle < 0] += 360
            mask = np.abs(angle - radial_angle) < angle_threshold
        elif direction == "radial_out":
            center_y, center_x = height // 2, width // 2
            radial_angle = np.arctan2(y - center_y, x - center_x) * 180 / np.pi
            radial_angle[radial_angle < 0] += 360
            mask = np.abs(angle - (radial_angle + 180) % 360) < angle_threshold
        elif direction == "clockwise":
            center_y, center_x = height // 2, width // 2
            radial_angle = np.arctan2(y - center_y, x - center_x) * 180 / np.pi
            radial_angle[radial_angle < 0] += 360
            mask = np.abs(angle - (radial_angle + 90) % 360) < angle_threshold
        elif direction == "counterclockwise":
            center_y, center_x = height // 2, width // 2
            radial_angle = np.arctan2(y - center_y, x - center_x) * 180 / np.pi
            radial_angle[radial_angle < 0] += 360
            mask = np.abs(angle - (radial_angle - 90) % 360) < angle_threshold

        mask = mask.astype(float)
        mask *= flow_magnitude

        if blur_radius > 0:
            mask = apply_blur(mask, blur_radius)

        mask = normalize_array(mask)

        if invert:
            mask = 1 - mask

        return np.clip(mask * strength, 0, 1)

    def apply_direction_mask(self, masks, images, strength, flow_method, flow_threshold, magnitude_threshold, direction, angle_threshold, blur_radius, invert, **kwargs):
        return super().main_function(masks, images, strength, flow_method, flow_threshold, magnitude_threshold, direction=direction, angle_threshold=angle_threshold, blur_radius=blur_radius, invert=invert, **kwargs)  

@apply_tooltips
class OpticalFlowParticleSystem(OpticalFlowMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "num_particles": ("INT", {"default": 1000, "min": 100, "max": 10000, "step": 100}),
                "particle_size": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "particle_color": ("STRING", {"default": "#FFFFFF"}),
                "particle_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "flow_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "particle_lifetime": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "initial_velocity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_particle_system"

    def __init__(self):
        super().__init__()
        self.particles = None

    def find_mask_edges(self, mask):
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        edges = cv2.Canny(mask_np.astype(np.uint8) * 255, 100, 200)
        return np.column_stack(np.where(edges > 0))

    def emit_particles(self, num_particles, mask, initial_velocity):
        edge_positions = self.find_mask_edges(mask)
        if len(edge_positions) > 0:
            indices = np.random.choice(len(edge_positions), num_particles, replace=True)
            positions = edge_positions[indices].astype(float)
            
            # Calculate outward directions
            center = np.mean(positions, axis=0)
            directions = positions - center
            directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
            
            velocities = directions * initial_velocity
            lifetimes = np.full(num_particles, self.particle_lifetime)
            
            # Return [y, x, vy, vx, lifetime]
            return np.column_stack((positions, velocities, lifetimes))
        return np.array([])

    def update_particles(self, flow, mask, height, width, flow_multiplier):
        if self.particles is None or len(self.particles) == 0:
            return

        flow_np = flow.cpu().numpy() if isinstance(flow, torch.Tensor) else flow

        # Clip particle positions to image boundaries
        np.clip(self.particles[:, 1], 0, width - 1, out=self.particles[:, 1])  # x
        np.clip(self.particles[:, 0], 0, height - 1, out=self.particles[:, 0])  # y

        # Update particle positions based on their velocity and optical flow
        flow_at_particles = flow_np[self.particles[:, 0].astype(int), self.particles[:, 1].astype(int)]
        self.particles[:, :2] += self.particles[:, 2:4] + flow_at_particles * flow_multiplier
        self.particles[:, 4] -= 1  # Decrease lifetime

        # Clip updated positions to image boundaries
        np.clip(self.particles[:, 1], 0, width - 1, out=self.particles[:, 1])  # x
        np.clip(self.particles[:, 0], 0, height - 1, out=self.particles[:, 0])  # y

        # Remove particles that have expired or reached the image boundaries
        self.particles = self.particles[
            (self.particles[:, 4] > 0) & 
            (self.particles[:, 1] > 0) & (self.particles[:, 1] < width - 1) &
            (self.particles[:, 0] > 0) & (self.particles[:, 0] < height - 1)
        ]

    def draw_particles(self, frame, particle_size, particle_color, particle_opacity):
        particle_color = tuple(int(particle_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to RGB
        for y, x, _, _, lifetime in self.particles:
            opacity = (lifetime / self.particle_lifetime) * particle_opacity
            cv2.circle(frame, (int(x), int(y)), particle_size, particle_color, -1)

        return cv2.addWeighted(frame, 1 - particle_opacity, np.zeros_like(frame), particle_opacity, 0)

    def generate_particle_system(self, masks, images, strength, flow_method, flow_threshold, magnitude_threshold, num_particles, particle_size, particle_color, particle_opacity, flow_multiplier, particle_lifetime, initial_velocity, **kwargs):
        results = []
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        images_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
        height, width = images_np[0].shape[:2]
        self.particle_lifetime = particle_lifetime
        self.particles = np.array([])

        for i in range(len(images_np) - 1):
            frame1 = (images_np[i] * 255).astype(np.uint8)
            frame2 = (images_np[i + 1] * 255).astype(np.uint8)

            flow = calculate_optical_flow(frame1, frame2, flow_method)

            # Emit new particles
            new_particles = self.emit_particles(num_particles // particle_lifetime, masks_np[i], initial_velocity)
            self.particles = np.vstack([self.particles, new_particles]) if self.particles.size > 0 else new_particles

            self.update_particles(flow, masks_np[i], height, width, flow_multiplier)

            particle_frame = np.zeros((height, width, 3), dtype=np.uint8)
            particle_frame = self.draw_particles(particle_frame, particle_size, particle_color, particle_opacity)

            results.append(particle_frame)

        return (torch.from_numpy(np.array(results) / 255.0).float(),)

    # Dummy implementation of apply_flow_mask to satisfy the abstract base class
    def apply_flow_mask(self, mask: np.ndarray, flow_magnitude: np.ndarray, flow: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        return mask  