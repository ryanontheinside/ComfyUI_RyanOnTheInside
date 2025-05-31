from ... import RyanOnTheInside 
import cv2
import numpy as np
import  torch
from scipy.spatial.distance import cdist
import math
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os
import random
import folder_paths

class EffectVisualizer(RyanOnTheInside):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "feature": ("FEATURE",),
                "text_color": ("STRING", {"default": "(255,255,255)"}),
                "font_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "label_prefix": ("STRING", {"default": ""}),
                "label_x_offset": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1}),
                "label_y_offset": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "RyanOnTheInside/FlexFeatures/Utilities/Previews"

    def visualize(self, video_frames, feature, text_color, font_scale, label_prefix, label_x_offset, label_y_offset):
        text_color = self.parse_color(text_color)
        output_frames = []
        padding = 10  # Padding from the edges

        for frame_index in range(len(video_frames)):
            frame = video_frames[frame_index].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display feature name and value on the frame
            feature_value = feature.get_value_at_frame(frame_index)
            prefix = f"{label_prefix} " if label_prefix else ""
            text = f"{prefix}{feature.name}: {feature_value:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = label_x_offset
            text_y = label_y_offset + text_size[1]

            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)

        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)

    def parse_color(self, color_string):
        return tuple(map(int, color_string.strip("()").split(",")))


class ProximityVisualizer(EffectVisualizer):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "anchor_locations": ("LOCATION",),
                "query_locations": ("LOCATION",),
                "anchor_color": ("STRING", {"default": "(255,0,0)"}),
                "query_color": ("STRING", {"default": "(0,255,0)"}),
                "line_color": ("STRING", {"default": "(0,0,255)"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_proximity"

    def visualize_proximity(self, video_frames, anchor_locations, query_locations, feature, 
                            anchor_color, query_color, line_color, text_color, font_scale,
                            label_prefix, label_x_offset, label_y_offset):
        anchor_color = self.parse_color(anchor_color)
        query_color = self.parse_color(query_color)
        line_color = self.parse_color(line_color)
        text_color = self.parse_color(text_color)

        output_frames = []
        height, width = video_frames.shape[1:3]  # Extract height and width from video_frames

        # Calculate the frame diagonal from the video frames
        frame_diagonal = np.sqrt(width**2 + height**2)
        scale_factor = frame_diagonal / feature.frame_diagonal

        for frame_index in range(len(video_frames)):
            frame = video_frames[frame_index].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            anchor = anchor_locations[frame_index]
            query = query_locations[frame_index]

            # Draw all anchor and query points (using only x and y coordinates)
            for point in anchor:
                scaled_point = (int(point[0] * scale_factor), int(point[1] * scale_factor))
                cv2.circle(frame, scaled_point, 2, anchor_color, -1)
            for point in query:
                scaled_point = (int(point[0] * scale_factor), int(point[1] * scale_factor))
                cv2.circle(frame, scaled_point, 2, query_color, -1)

            # Find the closest pair of points
            if len(anchor) > 0 and len(query) > 0:
                distances = cdist(anchor.points[:, :2], query.points[:, :2])  # Use only x and y for distance calculation
                min_idx = np.unravel_index(distances.argmin(), distances.shape)
                closest_anchor = anchor[min_idx[0]]
                closest_query = query[min_idx[1]]

                # Draw line between closest points (using only x and y coordinates)
                cv2.line(frame, 
                         (int(closest_anchor[0] * scale_factor), int(closest_anchor[1] * scale_factor)), 
                         (int(closest_query[0] * scale_factor), int(closest_query[1] * scale_factor)), 
                         line_color, 2)

                # Display coordinates of closest points
                anchor_text = f"Anchor: ({closest_anchor[0] * scale_factor:.2f}, {closest_anchor[1] * scale_factor:.2f}"
                query_text = f"Query: ({closest_query[0] * scale_factor:.2f}, {closest_query[1] * scale_factor:.2f}"
                
                if closest_anchor.shape[0] > 2 and closest_query.shape[0] > 2:
                    anchor_text += f", {closest_anchor[2]:.2f})"
                    query_text += f", {closest_query[2]:.2f})"
                else:
                    anchor_text += ") [2D]"
                    query_text += ") [2D]"

                cv2.putText(frame, anchor_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
                cv2.putText(frame, query_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

            # Add proximity value to the frame
            proximity_value = feature.get_value_at_frame(frame_index)
            cv2.putText(frame, f"Proximity: {proximity_value:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)

        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)



class PitchVisualizer(EffectVisualizer):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "feature": ("FEATURE",),
                "text_color": ("STRING", {"default": "(255,255,255)"}),
                "font_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "label_prefix": ("STRING", {"default": ""}),
                "label_x_offset": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1}),
                "label_y_offset": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_pitch"

    def visualize_pitch(self, video_frames, feature, text_color, font_scale, label_prefix, label_x_offset, label_y_offset):
        text_color = self.parse_color(text_color)
        output_frames = []
        padding = 10  # Padding from the edges

        # Ensure video_frames is BHWC
        if video_frames.shape[-1] != 3:
            video_frames = video_frames.permute(0, 2, 3, 1)

        for frame_index in range(video_frames.shape[0]):
            frame = video_frames[frame_index].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Get pitch feature data for the current frame
            pitch_data = feature.get_pitch_feature(frame_index)

            # Convert pitch to approximate note
            note = feature.pitch_to_note(pitch_data['actual_pitch'])
            smoothed_note = feature.pitch_to_note(pitch_data['smoothed_pitch'])

            # Display feature values on the frame
            texts = [
                f"Original: {pitch_data['original']:.2f}",
                f"Normalized: {pitch_data['normalized']:.2f}",
                f"Actual Pitch: {pitch_data['actual_pitch']:.2f} Hz",
                f"Note: {note}",
                f"Smoothed Pitch: {pitch_data['smoothed_pitch']:.2f} Hz",
                f"Smoothed Note: {smoothed_note}"
            ]

            for i, text in enumerate(texts):
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                text_x = label_x_offset
                text_y = label_y_offset + (i + 1) * (text_size[1] + 10)  # Add some vertical spacing

                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)

        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)


class PreviewFeature(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexFeatures/Utilities/Previews"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True

    def preview(self, feature, prompt=None, extra_pnginfo=None):
        width=960
        height=540
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Calculate actual min and max from the values
        actual_min = min(values)
        actual_max = max(values)
        
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.style.use('dark_background')
        
        plt.plot(values, color='dodgerblue', linewidth=2)
        
        plt.xlabel('Frame', color='white', fontsize=14)
        plt.ylabel('Value', color='white', fontsize=14)
        
        plt.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        plt.tick_params(axis='both', colors='white', labelsize=12)
        
        max_ticks = 10
        step = max(1, len(values) // max_ticks)
        x_ticks = range(0, len(values), step)
        plt.xticks(x_ticks, [str(x) for x in x_ticks])
        
        # Use actual min/max values with padding
        y_range = actual_max - actual_min
        if y_range == 0:  # Handle constant value case
            y_range = 1.0
            padding = 0.1
        else:
            padding = 0.05
        plt.ylim(actual_min - padding*y_range, actual_max + padding*y_range)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_color('white')
        plt.gca().spines['left'].set_color('white')
        
        plt.title(f'Feature: {feature.name}', color='white', fontsize=16)
        
        plt.tight_layout(pad=0.5)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='black', edgecolor='none')
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        plt.close()  
        buf.close()  
        
        # Save the image to a temporary file like PreviewImage does
        output_dir = folder_paths.get_temp_directory()
        type = "temp"
        prefix = "feature_preview_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(prefix, output_dir, img_tensor.shape[2], img_tensor.shape[1])
        
        # Save the image
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i[0], 0, 255).astype(np.uint8))
        file = f"{filename}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), compress_level=1)
        
        results = [{
            "filename": file,
            "subfolder": subfolder,
            "type": type
        }]
        
        return ({"ui": {"images": results}})

class AnimatedFeaturePreview(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexFeatures/Utilities/Previews"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "feature": ("FEATURE",),
                "window_size": ("INT", {"default": 60, "min": 10, "max": 3000, "step": 1}),
                "top_label": ("STRING", {"default": "Max"}),
                "bottom_label": ("STRING", {"default": "Min"}),
                "width": ("INT", {"default": 960, "min": 480, "max": 1920, "step": 1}),
                "height": ("INT", {"default": 540, "min": 270, "max": 1080, "step": 1}),
                "low_color": ("STRING", {"default": "(255,100,100)"}),  # Red for low values
                "high_color": ("STRING", {"default": "(100,255,100)"}),  # Green for high values
                "title_override": ("STRING", {"default": ""}),  # Override for feature name in title
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "animate_feature"

    def parse_color(self, color_string):
        """Parse color string to tuple"""
        return tuple(map(int, color_string.strip("()").split(",")))

    def interpolate_color(self, color1, color2, factor):
        """Interpolate between two colors based on factor (0.0 to 1.0)"""
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

    def draw_text_with_background(self, frame, text, position, font, font_scale, text_color, thickness=2, bg_color=(0, 0, 0), padding=8):
        """Draw text with a background rectangle for better visibility"""
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x, y = position
        
        # Draw background rectangle with some padding
        bg_rect_start = (x - padding, y - text_size[1] - padding)
        bg_rect_end = (x + text_size[0] + padding, y + padding)
        
        # Add subtle transparency effect by blending
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_rect_start, bg_rect_end, bg_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, bg_rect_start, bg_rect_end, (64, 64, 64), 1)
        
        # Draw text
        cv2.putText(frame, text, position, font, font_scale, text_color, thickness)

    def draw_gradient_background(self, frame, start_color, end_color):
        """Draw a subtle gradient background"""
        height, width = frame.shape[:2]
        for y in range(height):
            factor = y / height
            color = self.interpolate_color(start_color, end_color, factor)
            frame[y, :] = color

    def animate_feature(self, feature, window_size, top_label, bottom_label, width, height, low_color, high_color, title_override):
        low_color = self.parse_color(low_color)
        high_color = self.parse_color(high_color)
        
        values = [feature.get_value_at_frame(i) for i in range(feature.frame_count)]
        
        # Calculate actual min and max from the values
        actual_min = min(values)
        actual_max = max(values)
        
        # Handle constant value case
        if actual_max == actual_min:
            actual_max = actual_min + 1.0
            
        y_range = actual_max - actual_min
        padding = 0.05 * y_range
        y_min = actual_min - padding
        y_max = actual_max + padding
        
        # Graph area parameters with better proportions for labels above/below
        margin_x = 80
        margin_y = 120  # More space for labels above/below
        graph_width = width - 2 * margin_x
        graph_height = height - 2 * margin_y
        graph_x = margin_x
        graph_y = margin_y
        
        output_frames = []
        
        for frame_index in range(feature.frame_count):
            # Create frame with gradient background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            self.draw_gradient_background(frame, (15, 15, 25), (25, 25, 35))
            
            # Determine the window for this frame
            window_start = max(0, frame_index - window_size // 2)
            window_end = min(feature.frame_count, window_start + window_size)
            
            # Adjust window_start if we're near the end
            if window_end - window_start < window_size and window_start > 0:
                window_start = max(0, window_end - window_size)
            
            # Draw enhanced grid
            grid_color = (45, 45, 55)  # Subtle grid
            major_grid_color = (65, 65, 75)  # Slightly more visible for major lines
            
            # Horizontal grid lines
            for i in range(9):  # More grid lines for precision
                y = graph_y + (i * graph_height // 8)
                color = major_grid_color if i % 2 == 0 else grid_color
                cv2.line(frame, (graph_x, y), (graph_x + graph_width, y), color, 1)
            
            # Vertical grid lines
            for i in range(13):  # More vertical lines
                x = graph_x + (i * graph_width // 12)
                color = major_grid_color if i % 3 == 0 else grid_color
                cv2.line(frame, (x, graph_y), (x, graph_y + graph_height), color, 1)
            
            # Draw graph border with gradient effect
            border_color = (120, 120, 140)
            cv2.rectangle(frame, (graph_x-2, graph_y-2), (graph_x + graph_width+2, graph_y + graph_height+2), border_color, 3)
            cv2.rectangle(frame, (graph_x-1, graph_y-1), (graph_x + graph_width+1, graph_y + graph_height+1), (200, 200, 220), 1)
            
            # Get the values for the current window
            window_values = values[window_start:window_end]
            window_frames = list(range(window_start, window_end))
            
            # Convert values to screen coordinates and draw enhanced line with smoother interpolation
            if len(window_values) > 1:
                points = []
                point_values = []
                for i, (f, v) in enumerate(zip(window_frames, window_values)):
                    # X coordinate based on frame position in window
                    x_ratio = (f - window_start) / window_size
                    x = int(graph_x + x_ratio * graph_width)
                    
                    # Y coordinate based on value (inverted because screen Y increases downward)
                    y_ratio = (v - y_min) / (y_max - y_min)
                    y = int(graph_y + graph_height - y_ratio * graph_height)
                    
                    points.append((x, y))
                    point_values.append(v)
                
                # Draw the line with smooth gradient colors
                for i in range(len(points) - 1):
                    # Create multiple segments for smoother color transition
                    start_point = points[i]
                    end_point = points[i + 1]
                    start_value = point_values[i]
                    end_value = point_values[i + 1]
                    
                    # Number of segments for smooth color transition
                    num_segments = max(5, abs(end_point[0] - start_point[0]) // 2)
                    
                    for seg in range(num_segments):
                        seg_ratio = seg / num_segments
                        next_seg_ratio = (seg + 1) / num_segments
                        
                        # Interpolate position
                        seg_x = int(start_point[0] + (end_point[0] - start_point[0]) * seg_ratio)
                        seg_y = int(start_point[1] + (end_point[1] - start_point[1]) * seg_ratio)
                        next_x = int(start_point[0] + (end_point[0] - start_point[0]) * next_seg_ratio)
                        next_y = int(start_point[1] + (end_point[1] - start_point[1]) * next_seg_ratio)
                        
                        # Interpolate value for color
                        seg_value = start_value + (end_value - start_value) * (seg_ratio + next_seg_ratio) / 2
                        color_factor = (seg_value - actual_min) / (actual_max - actual_min) if actual_max != actual_min else 0.5
                        line_color = self.interpolate_color(low_color, high_color, color_factor)
                        
                        # Draw segment with glow effect
                        cv2.line(frame, (seg_x, seg_y), (next_x, next_y), (line_color[0]//3, line_color[1]//3, line_color[2]//3), 5)  # Glow
                        cv2.line(frame, (seg_x, seg_y), (next_x, next_y), line_color, 3)  # Main line
                
                # Draw the current position dot with enhanced styling
                current_value = values[frame_index]
                x_ratio = (frame_index - window_start) / window_size
                x = int(graph_x + x_ratio * graph_width)
                y_ratio = (current_value - y_min) / (y_max - y_min)
                y = int(graph_y + graph_height - y_ratio * graph_height)
                
                # Calculate dot color based on current value
                color_factor = (current_value - actual_min) / (actual_max - actual_min) if actual_max != actual_min else 0.5
                dot_color = self.interpolate_color(low_color, high_color, color_factor)
                
                # Draw dot with glow effect
                cv2.circle(frame, (x, y), 12, (dot_color[0]//2, dot_color[1]//2, dot_color[2]//2), -1)  # Outer glow
                cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)  # White outline
                cv2.circle(frame, (x, y), 6, dot_color, -1)     # Colored fill
            
            # Enhanced text rendering
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)
            
            # Labels positioned next to the y-axis
            # Top label (left side, aligned with top of graph)
            self.draw_text_with_background(frame, top_label, (15, graph_y + 20), font, 0.9, text_color, 2, (20, 20, 40), 8)
            
            # Bottom label (left side, aligned with bottom of graph)
            self.draw_text_with_background(frame, bottom_label, (15, graph_y + graph_height - 10), font, 0.9, text_color, 2, (20, 20, 40), 8)
            
            # Enhanced title (centered at top)
            title = title_override if title_override.strip() else feature.name
            title_size = cv2.getTextSize(title, font, 1.2, 3)[0]
            title_x = (width - title_size[0]) // 2
            self.draw_text_with_background(frame, title, (title_x, 40), font, 1.2, text_color, 3, (20, 20, 40), 12)
            
            # Current value (top right, no crowding)
            current_value = values[frame_index]
            value_text = f'Value: {current_value:.4f}'
            value_size = cv2.getTextSize(value_text, font, 0.8, 2)[0]
            value_pos = (width - value_size[0] - 20, 70)  # Moved down to avoid title
            
            # Color the value text based on the current value
            color_factor = (current_value - actual_min) / (actual_max - actual_min) if actual_max != actual_min else 0.5
            value_color = self.interpolate_color(low_color, high_color, color_factor)
            self.draw_text_with_background(frame, value_text, value_pos, font, 0.8, value_color, 2, (20, 20, 40), 8)
            
            # Frame info (bottom left - no crowding)
            frame_info = f'Frame: {frame_index + 1}/{feature.frame_count}'
            self.draw_text_with_background(frame, frame_info, (20, height - 30), font, 0.7, text_color, 2, (20, 20, 40), 6)
            
            # Window range info (bottom right, properly spaced)
            range_text = f'Window: {window_start}-{window_end-1}'
            range_size = cv2.getTextSize(range_text, font, 0.6, 1)[0]
            range_pos = (width - range_size[0] - 20, height - 30)  # Same height as frame info
            self.draw_text_with_background(frame, range_text, range_pos, font, 0.6, (180, 180, 180), 1, (20, 20, 40), 4)
            
            output_frames.append(frame)
        
        # Convert to tensor
        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)