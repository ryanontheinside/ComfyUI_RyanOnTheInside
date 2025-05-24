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