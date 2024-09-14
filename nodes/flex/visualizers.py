from ... import RyanOnTheInside 
import cv2
import numpy as np
import  torch
from scipy.spatial.distance import cdist
import math

class EffectVisualizer(RyanOnTheInside):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "feature": ("FEATURE",),
                "text_color": ("STRING", {"default": "(255,255,255)"}),
                "font_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "RyanOnTheInside/FlexFeatures/EffectVisualizers"

    def visualize(self, video_frames, feature, text_color, font_scale):
        text_color = self.parse_color(text_color)
        output_frames = []
        padding = 10  # Padding from the edges

        for frame_index in range(len(video_frames)):
            frame = video_frames[frame_index].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display feature name and value on the frame
            feature_value = feature.get_value_at_frame(frame_index)
            text = f"{feature.name}: {feature_value:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = padding
            text_y = padding + text_size[1]

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
                            anchor_color, query_color, line_color, text_color, font_scale):
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_pitch"

    def visualize_pitch(self, video_frames, feature, text_color, font_scale):
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
                text_x = padding
                text_y = padding + (i + 1) * (text_size[1] + 10)  # Add some vertical spacing

                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)

        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)
