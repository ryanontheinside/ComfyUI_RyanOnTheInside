from ... import RyanOnTheInside, ProgressMixin
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

        actual_min = min(values)
        actual_max = max(values)
        if actual_max == actual_min:
            actual_max = actual_min + 1.0
        y_range = actual_max - actual_min
        padding = 0.05 * y_range
        y_min = actual_min - padding
        y_max = actual_max + padding

        # Layout — tighter margins matching float preview
        margin_x = 80
        margin_top = 80
        margin_bottom = 60
        graph_width = width - 2 * margin_x
        graph_height = height - margin_top - margin_bottom
        graph_x = margin_x
        graph_y = margin_top

        font = cv2.FONT_HERSHEY_SIMPLEX
        output_frames = []

        for frame_index in range(feature.frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            self.draw_gradient_background(frame, (15, 15, 25), (25, 25, 35))

            # Window bounds
            window_start = max(0, frame_index - window_size // 2)
            window_end = min(feature.frame_count, window_start + window_size)
            if window_end - window_start < window_size and window_start > 0:
                window_start = max(0, window_end - window_size)

            # Grid
            grid_color = (45, 45, 55)
            major_grid_color = (65, 65, 75)

            for i in range(9):
                y = graph_y + (i * graph_height // 8)
                color = major_grid_color if i % 2 == 0 else grid_color
                cv2.line(frame, (graph_x, y), (graph_x + graph_width, y), color, 1)

            for i in range(13):
                x = graph_x + (i * graph_width // 12)
                color = major_grid_color if i % 3 == 0 else grid_color
                cv2.line(frame, (x, graph_y), (x, graph_y + graph_height), color, 1)

            # Border
            cv2.rectangle(frame, (graph_x - 2, graph_y - 2),
                          (graph_x + graph_width + 2, graph_y + graph_height + 2), (120, 120, 140), 3)
            cv2.rectangle(frame, (graph_x - 1, graph_y - 1),
                          (graph_x + graph_width + 1, graph_y + graph_height + 1), (200, 200, 220), 1)

            # Y-axis numeric labels
            for i in range(5):
                y_val = y_min + (y_max - y_min) * (1.0 - i / 4.0)
                y_pos = graph_y + (i * graph_height // 4)
                label_text = f"{y_val:.2f}"
                label_size = cv2.getTextSize(label_text, font, 0.45, 1)[0]
                cv2.putText(frame, label_text, (graph_x - label_size[0] - 8, y_pos + 4),
                            font, 0.45, (140, 140, 160), 1)

            # Top/bottom labels (left side, aligned with Y-axis)
            if top_label:
                tl_color = self.interpolate_color(low_color, high_color, 1.0)
                tl_size = cv2.getTextSize(top_label, font, 0.55, 1)[0]
                cv2.putText(frame, top_label, (graph_x - tl_size[0] - 8, graph_y - 10),
                            font, 0.55, tl_color, 1)
            if bottom_label:
                bl_color = self.interpolate_color(low_color, high_color, 0.0)
                bl_size = cv2.getTextSize(bottom_label, font, 0.55, 1)[0]
                cv2.putText(frame, bottom_label, (graph_x - bl_size[0] - 8, graph_y + graph_height + 20),
                            font, 0.55, bl_color, 1)

            # Draw line
            window_values = values[window_start:window_end]
            window_frames = list(range(window_start, window_end))

            if len(window_values) > 1:
                points = []
                point_values = []
                for i, (f, v) in enumerate(zip(window_frames, window_values)):
                    x_ratio = (f - window_start) / window_size
                    x = int(graph_x + x_ratio * graph_width)
                    y_ratio = (v - y_min) / (y_max - y_min)
                    y = int(graph_y + graph_height - y_ratio * graph_height)
                    points.append((x, y))
                    point_values.append(v)

                # Draw line with gradient colors
                for i in range(len(points) - 1):
                    start_point = points[i]
                    end_point = points[i + 1]
                    start_value = point_values[i]
                    end_value = point_values[i + 1]

                    num_segments = max(5, abs(end_point[0] - start_point[0]) // 2)
                    for seg in range(num_segments):
                        seg_ratio = seg / num_segments
                        next_seg_ratio = (seg + 1) / num_segments

                        seg_x = int(start_point[0] + (end_point[0] - start_point[0]) * seg_ratio)
                        seg_y = int(start_point[1] + (end_point[1] - start_point[1]) * seg_ratio)
                        next_x = int(start_point[0] + (end_point[0] - start_point[0]) * next_seg_ratio)
                        next_y = int(start_point[1] + (end_point[1] - start_point[1]) * next_seg_ratio)

                        seg_value = start_value + (end_value - start_value) * (seg_ratio + next_seg_ratio) / 2
                        color_factor = (seg_value - actual_min) / (actual_max - actual_min) if actual_max != actual_min else 0.5
                        line_color = self.interpolate_color(low_color, high_color, color_factor)

                        cv2.line(frame, (seg_x, seg_y), (next_x, next_y),
                                 (line_color[0] // 4, line_color[1] // 4, line_color[2] // 4), 5)
                        cv2.line(frame, (seg_x, seg_y), (next_x, next_y), line_color, 2)

                # Current position dot
                current_value = values[frame_index]
                x_ratio = (frame_index - window_start) / window_size
                x = int(graph_x + x_ratio * graph_width)
                y_ratio = (current_value - y_min) / (y_max - y_min)
                y = int(graph_y + graph_height - y_ratio * graph_height)

                color_factor = (current_value - actual_min) / (actual_max - actual_min) if actual_max != actual_min else 0.5
                dot_color = self.interpolate_color(low_color, high_color, color_factor)

                cv2.circle(frame, (x, y), 8, (dot_color[0] // 2, dot_color[1] // 2, dot_color[2] // 2), -1)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
                cv2.circle(frame, (x, y), 4, dot_color, -1)

            # Title (centered at top)
            title = title_override if title_override.strip() else feature.name
            title_size = cv2.getTextSize(title, font, 1.0, 2)[0]
            title_x = (width - title_size[0]) // 2
            overlay = frame.copy()
            cv2.rectangle(overlay, (title_x - 12, 8), (title_x + title_size[0] + 12, 48), (20, 20, 40), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, title, (title_x, 38), font, 1.0, (255, 255, 255), 2)

            # Current value (top right)
            current_value = values[frame_index]
            color_factor = (current_value - actual_min) / (actual_max - actual_min) if actual_max != actual_min else 0.5
            value_color = self.interpolate_color(low_color, high_color, color_factor)
            value_text = f"Value: {current_value:.4f}"
            value_size = cv2.getTextSize(value_text, font, 0.6, 1)[0]
            cv2.putText(frame, value_text, (width - value_size[0] - 20, 38),
                        font, 0.6, value_color, 1)

            # Frame info (bottom left)
            frame_info = f"Frame: {frame_index + 1}/{feature.frame_count}"
            cv2.putText(frame, frame_info, (20, height - 12), font, 0.55, (180, 180, 180), 1)

            # Window range (bottom right)
            range_text = f"Window: {window_start}-{window_end - 1}"
            range_size = cv2.getTextSize(range_text, font, 0.5, 1)[0]
            cv2.putText(frame, range_text, (width - range_size[0] - 20, height - 12),
                        font, 0.5, (140, 140, 140), 1)

            output_frames.append(frame)

        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)

class AnimatedFloatPreview(RyanOnTheInside, ProgressMixin):
    """Animated preview for multiple float sequences with legend and sliding window."""

    CATEGORY = "RyanOnTheInside/FlexFeatures/Utilities/Previews"
    DESCRIPTION = "Plots up to 6 float sequences on a single animated graph with a sliding window, legend, and per-line colors. Accepts scalar or list-of-float inputs."

    # Distinct line colors (BGR for cv2)
    LINE_COLORS = [
        (100, 100, 255),  # Red
        (100, 255, 100),  # Green
        (255, 100, 100),  # Blue
        (0, 220, 220),    # Yellow
        (220, 100, 220),  # Magenta
        (220, 220, 100),  # Cyan
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "window_size": ("INT", {"default": 60, "min": 10, "max": 3000, "step": 1}),
                "width": ("INT", {"default": 960, "min": 480, "max": 1920, "step": 1}),
                "height": ("INT", {"default": 540, "min": 270, "max": 1080, "step": 1}),
                "title": ("STRING", {"default": "Float Preview"}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional audio input — renders a spectrogram behind the graph"}),
                "audio_brightness": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "Brightness of the background spectrogram (lower = more subtle)"}),
                "float_1": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "label_1": ("STRING", {"default": "Float 1"}),
                "float_2": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "label_2": ("STRING", {"default": "Float 2"}),
                "float_3": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "label_3": ("STRING", {"default": "Float 3"}),
                "float_4": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "label_4": ("STRING", {"default": "Float 4"}),
                "float_5": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "label_5": ("STRING", {"default": "Float 5"}),
                "float_6": ("FLOAT", {"default": 0.0, "forceInput": True}),
                "label_6": ("STRING", {"default": "Float 6"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "animate_floats"

    def _parse_input(self, value, frame_count):
        """Convert scalar or list input to a list of length frame_count."""
        if isinstance(value, list):
            if len(value) == frame_count:
                return [float(v) for v in value]
            # Resample to frame_count
            result = []
            src_len = len(value)
            for i in range(frame_count):
                idx = round(i * (src_len - 1) / max(frame_count - 1, 1)) if frame_count > 1 else 0
                idx = max(0, min(idx, src_len - 1))
                result.append(float(value[idx]))
            return result
        else:
            return [float(value)] * frame_count

    @staticmethod
    def _compute_band_energy(audio, frame_count):
        """Precompute smoothed low/mid/high frequency energy per frame.
        Returns array [frame_count, 3] with values in [0, 1]."""
        waveform = audio["waveform"].squeeze(0).mean(dim=0).cpu().numpy()
        sr = audio["sample_rate"]
        total_samples = len(waveform)
        samples_per_frame = total_samples / max(frame_count, 1)
        fft_size = max(1024, int(samples_per_frame * 2))

        # Frequency bin boundaries: low <300Hz, mid 300-3000Hz, high >3000Hz
        freq_per_bin = sr / fft_size
        low_end = int(300 / freq_per_bin)
        mid_end = int(3000 / freq_per_bin)

        bands = np.zeros((frame_count, 3), dtype=np.float32)
        for i in range(frame_count):
            center = int(i * samples_per_frame)
            start = max(0, center - fft_size // 2)
            end = min(total_samples, start + fft_size)
            chunk = waveform[start:end]
            if len(chunk) < fft_size:
                chunk = np.pad(chunk, (0, fft_size - len(chunk)))
            spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
            bands[i, 0] = np.mean(spectrum[1:max(2, low_end)])
            bands[i, 1] = np.mean(spectrum[low_end:max(low_end + 1, mid_end)])
            bands[i, 2] = np.mean(spectrum[mid_end:max(mid_end + 1, len(spectrum))])

        # Normalize each band independently
        for b in range(3):
            bmax = bands[:, b].max()
            if bmax > 0:
                bands[:, b] /= bmax

        # Temporal smoothing (EMA)
        smoothed = np.zeros_like(bands)
        smoothed[0] = bands[0]
        alpha = 0.3
        for i in range(1, frame_count):
            smoothed[i] = alpha * bands[i] + (1 - alpha) * smoothed[i - 1]
        for b in range(3):
            bmax = smoothed[:, b].max()
            if bmax > 0:
                smoothed[:, b] /= bmax
        return smoothed

    def animate_floats(self, window_size, width, height, title,
                       audio=None, audio_brightness=0.3,
                       float_1=None, label_1="Float 1",
                       float_2=None, label_2="Float 2",
                       float_3=None, label_3="Float 3",
                       float_4=None, label_4="Float 4",
                       float_5=None, label_5="Float 5",
                       float_6=None, label_6="Float 6"):

        # Collect connected inputs and determine frame_count from longest list
        raw_inputs = [
            (float_1, label_1), (float_2, label_2), (float_3, label_3),
            (float_4, label_4), (float_5, label_5), (float_6, label_6),
        ]

        # Determine frame_count from the longest connected list
        frame_count = 1
        for val, _ in raw_inputs:
            if isinstance(val, list) and len(val) > frame_count:
                frame_count = len(val)

        series = []  # list of (label, values_list, color)
        for i, (val, label) in enumerate(raw_inputs):
            if val is None:
                continue
            values = self._parse_input(val, frame_count)
            series.append((label, values, self.LINE_COLORS[i % len(self.LINE_COLORS)]))

        if not series:
            # Nothing connected — return black frames
            blank = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
            return (torch.from_numpy(blank).float() / 255.0,)

        # Precompute audio band energy for reactive background
        band_energy = None
        if audio is not None:
            band_energy = self._compute_band_energy(audio, frame_count)

        # Calculate global min/max across all series
        all_values = [v for _, vals, _ in series for v in vals]
        global_min = min(all_values)
        global_max = max(all_values)
        if global_max == global_min:
            global_max = global_min + 1.0
        y_range = global_max - global_min
        padding = 0.05 * y_range
        y_min = global_min - padding
        y_max = global_max + padding

        # Layout
        margin_x = 80
        margin_top = 80
        legend_height = 30 * ((len(series) + 2) // 3)  # rows of 3 in legend
        margin_bottom = 60 + legend_height
        graph_width = width - 2 * margin_x
        graph_height = height - margin_top - margin_bottom
        graph_x = margin_x
        graph_y = margin_top

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Precompute ALL glow frames at once (vectorized)
        all_glow_frames = None
        if band_energy is not None:
            vert = np.linspace(0, 1, graph_height)[:, np.newaxis]
            ones_h = np.ones((1, graph_width))
            # Build 3 base masks [H, W, 3] and blur once each
            masks = []
            for shape_fn, colors in [
                (np.power(vert, 0.5) * ones_h,       (0.4, 0.12, 0.5)),   # low: purple bottom
                (np.exp(-8.0 * (vert - 0.5)**2) * ones_h, (0.15, 0.5, 0.45)), # mid: teal center
                (np.power(1.0 - vert, 0.8) * ones_h, (0.2, 0.35, 0.55)),  # high: cool top
            ]:
                m = np.zeros((graph_height, graph_width, 3), dtype=np.float32)
                m[:, :, 0] = shape_fn * colors[0]
                m[:, :, 1] = shape_fn * colors[1]
                m[:, :, 2] = shape_fn * colors[2]
                m = cv2.GaussianBlur(m, (0, 0), sigmaX=25, sigmaY=20)
                masks.append(m)
            # Stack masks: [3, H, W, 3]
            masks_arr = np.stack(masks, axis=0)
            # band_energy: [frame_count, 3] → einsum to get [frame_count, H, W, 3]
            # glow_frame[f] = sum_b(band_energy[f, b] * masks[b])
            all_glow_frames = np.einsum('fb,bhwc->fhwc', band_energy, masks_arr)
            all_glow_frames = np.clip(all_glow_frames * 255 * audio_brightness, 0, 255).astype(np.uint8)

        # Precompute background gradient
        bg_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            factor = y / height
            bg_frame[y, :] = (int(15 + 10 * factor), int(15 + 10 * factor), int(25 + 10 * factor))

        output_frames = []
        self.start_progress(frame_count, desc="Rendering Float Preview")

        for frame_index in range(frame_count):
            frame = bg_frame.copy()

            # Window bounds
            window_start = max(0, frame_index - window_size // 2)
            window_end = min(frame_count, window_start + window_size)
            if window_end - window_start < window_size and window_start > 0:
                window_start = max(0, window_end - window_size)

            # Audio-reactive ambient glow — just index into precomputed array
            if all_glow_frames is not None:
                graph_region = frame[graph_y:graph_y + graph_height, graph_x:graph_x + graph_width]
                cv2.add(graph_region, all_glow_frames[frame_index], dst=graph_region)
                frame[graph_y:graph_y + graph_height, graph_x:graph_x + graph_width] = graph_region

            # Grid
            grid_color = (45, 45, 55)
            major_grid_color = (65, 65, 75)

            for i in range(9):
                y = graph_y + (i * graph_height // 8)
                color = major_grid_color if i % 2 == 0 else grid_color
                cv2.line(frame, (graph_x, y), (graph_x + graph_width, y), color, 1)

            for i in range(13):
                x = graph_x + (i * graph_width // 12)
                color = major_grid_color if i % 3 == 0 else grid_color
                cv2.line(frame, (x, graph_y), (x, graph_y + graph_height), color, 1)

            # Border
            cv2.rectangle(frame, (graph_x - 2, graph_y - 2),
                          (graph_x + graph_width + 2, graph_y + graph_height + 2), (120, 120, 140), 3)
            cv2.rectangle(frame, (graph_x - 1, graph_y - 1),
                          (graph_x + graph_width + 1, graph_y + graph_height + 1), (200, 200, 220), 1)

            # Y-axis labels
            for i in range(5):
                y_val = y_min + (y_max - y_min) * (1.0 - i / 4.0)
                y_pos = graph_y + (i * graph_height // 4)
                label_text = f"{y_val:.2f}"
                label_size = cv2.getTextSize(label_text, font, 0.45, 1)[0]
                cv2.putText(frame, label_text, (graph_x - label_size[0] - 8, y_pos + 4),
                            font, 0.45, (140, 140, 160), 1)

            # Draw each series
            for label, values, color in series:
                window_values = values[window_start:window_end]

                if len(window_values) < 2:
                    continue

                points = []
                for i, v in enumerate(window_values):
                    f_idx = window_start + i
                    x_ratio = (f_idx - window_start) / max(window_size, 1)
                    x = int(graph_x + x_ratio * graph_width)
                    y_ratio = (v - y_min) / (y_max - y_min)
                    y = int(graph_y + graph_height - y_ratio * graph_height)
                    points.append((x, y))

                # Draw line segments
                for i in range(len(points) - 1):
                    # Glow
                    cv2.line(frame, points[i], points[i + 1],
                             (color[0] // 4, color[1] // 4, color[2] // 4), 5)
                    # Main line
                    cv2.line(frame, points[i], points[i + 1], color, 2)

                # Current position dot
                if window_start <= frame_index < window_end:
                    dot_idx = frame_index - window_start
                    if dot_idx < len(points):
                        px, py = points[dot_idx]
                        cv2.circle(frame, (px, py), 8,
                                   (color[0] // 2, color[1] // 2, color[2] // 2), -1)
                        cv2.circle(frame, (px, py), 5, (255, 255, 255), 1)
                        cv2.circle(frame, (px, py), 4, color, -1)

            # Title
            title_size = cv2.getTextSize(title, font, 1.0, 2)[0]
            title_x = (width - title_size[0]) // 2
            # Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (title_x - 12, 8), (title_x + title_size[0] + 12, 48), (20, 20, 40), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, title, (title_x, 38), font, 1.0, (255, 255, 255), 2)

            # Legend (below graph, arranged in rows of 3)
            legend_y_start = graph_y + graph_height + 25
            cols = 3
            col_width = graph_width // cols
            for i, (label, values, color) in enumerate(series):
                row = i // cols
                col = i % cols
                lx = graph_x + col * col_width
                ly = legend_y_start + row * 28

                # Color swatch
                cv2.rectangle(frame, (lx, ly - 8), (lx + 20, ly + 4), color, -1)
                cv2.rectangle(frame, (lx, ly - 8), (lx + 20, ly + 4), (200, 200, 200), 1)

                # Label + current value
                current_val = values[frame_index] if frame_index < len(values) else 0.0
                legend_text = f"{label}: {current_val:.3f}"
                cv2.putText(frame, legend_text, (lx + 28, ly + 4), font, 0.5, color, 1)

            # Frame info (bottom left)
            frame_info = f"Frame: {frame_index + 1}/{frame_count}"
            cv2.putText(frame, frame_info, (20, height - 12), font, 0.55, (180, 180, 180), 1)

            # Window range (bottom right)
            range_text = f"Window: {window_start}-{window_end - 1}"
            range_size = cv2.getTextSize(range_text, font, 0.5, 1)[0]
            cv2.putText(frame, range_text, (width - range_size[0] - 20, height - 12),
                        font, 0.5, (140, 140, 140), 1)

            output_frames.append(frame)
            self.update_progress()

        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "ProximityVisualizer": ProximityVisualizer,
    "EffectVisualizer": EffectVisualizer,
    "PitchVisualizer": PitchVisualizer,
    "PreviewFeature": PreviewFeature,
    "AnimatedFeaturePreview": AnimatedFeaturePreview,
    "AnimatedFloatPreview": AnimatedFloatPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProximityVisualizer": "Preview Proximity",
    "EffectVisualizer": "Preview FeatureEffect",
    "PitchVisualizer": "Preview Pitch",
}