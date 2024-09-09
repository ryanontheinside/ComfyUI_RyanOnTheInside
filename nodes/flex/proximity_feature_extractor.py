from .proximity_feature import Location,   ProximityFeature
from .feature_extractors import FirstFeature
from ... import RyanOnTheInside
from .feature_pipe import FeaturePipe
import numpy as np
import cv2
import  torch
from scipy.spatial.distance import cdist  # Add this import

class ProximityFeatureNode(FirstFeature):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.1}),
                "anchor_locations": ("LOCATION",),
                "query_locations": ("LOCATION",),
                "normalization_method": (["frame", "minmax"],),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    RETURN_NAMES = ("proximity_feature", "feature_pipe")
    FUNCTION = "create_feature"

    def create_feature(self, video_frames, frame_rate, anchor_locations, query_locations, normalization_method):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        frame_dimensions = (video_frames.shape[2], video_frames.shape[1])  # width, height
        
        proximity_feature = ProximityFeature(
            name="proximity_feature",
            anchor_locations=anchor_locations,
            query_locations=query_locations,
            frame_rate=frame_rate,
            frame_count=len(video_frames),
            frame_dimensions=frame_dimensions,
            normalization_method=normalization_method
        )

        # Ensure anchor and query points have the same number of columns
        anchor_dim = anchor_locations[0].points.shape[1]
        query_dim = query_locations[0].points.shape[1]

        if anchor_dim != query_dim:
            if anchor_dim == 3:
                for location in anchor_locations:
                    location.points = location.points[:, :2]
            elif query_dim == 3:
                for location in query_locations:
                    location.points = location.points[:, :2]
            print("Depth ignored. Depth must be included for all points or no points.")

        # self.start_progress(len(video_frames), desc="Calculating proximity")
        proximity_feature.extract()
        # self.end_progress()

        return (proximity_feature, feature_pipe)

class ProximityFeatureInput(RyanOnTheInside):
    CATEGORY="RyanOnTheInside/Proximity"

class LocationFromMask(ProximityFeatureInput):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "method": (["mask_boundary", "mask_center"],),
            },
            "optional": {
                "depth_maps": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LOCATION",)
    FUNCTION = "generate_locations"

    def generate_locations(self, masks, method, depth_maps=None):
        locations = []
        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            if np.sum(mask) == 0:  # Check if the mask is empty
                print(f"No mask found in frame {i}.")
                locations.append(Location(np.array([-1]), np.array([-1]), np.array([-1]) if depth_maps is not None else None))
                continue

            if method == "mask_boundary":
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    boundary = np.vstack(contours).squeeze()
                else:
                    boundary = np.array([[0, 0]])  # Fallback to a single point
            elif method == "mask_center":
                y, x = np.where(mask > 0.5)
                if len(x) > 0 and len(y) > 0:
                    center_x, center_y = np.mean(x), np.mean(y)
                    boundary = np.array([[center_x, center_y]])
                else:
                    boundary = np.array([[mask.shape[1] / 2, mask.shape[0] / 2]])

            if depth_maps is not None:
                depth_map = depth_maps[i].cpu().numpy()
                # Ensure boundary points are within the depth map dimensions
                valid_points = (boundary[:, 0] < depth_map.shape[1]) & (boundary[:, 1] < depth_map.shape[0])
                boundary = boundary[valid_points]
                
                if boundary.size == 0:  # Check if boundary is empty after filtering
                    print(f"No valid boundary points in frame {i}.")
                    locations.append(Location(np.array([-1]), np.array([-1]), np.array([-1])))
                    continue

                # Extract depth values for valid boundary points
                z = np.mean(depth_map[boundary[:, 1].astype(int), boundary[:, 0].astype(int)], axis=-1)
                
                location = Location(boundary[:, 0], boundary[:, 1], z)
            else:
                location = Location(boundary[:, 0], boundary[:, 1])
            
            locations.append(location)

        return (locations,)

class LocationFromPoint(ProximityFeatureInput):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x": ("FLOAT", {"default": 0.0,"min":0.0,"step":0.01}),
                "y": ("FLOAT", {"default": 0.0,"min":0.0,"step":0.01}),
                "z": ("FLOAT", {"default": 0.0,  "max":1.0, "min":0.0,"step":0.01}),
                "batch_count": ("INT", {"default": 1, "min": 1}),
            }
        }

    RETURN_TYPES = ("LOCATION",)
    FUNCTION = "generate_locations"

    def generate_locations(self, x, y, batch_count, z):
        locations = []
        for _ in range(batch_count):
            location = Location(np.array([x]), np.array([y]), np.array([z]))
            locations.append(location)

        return (locations,)

class LocationTransform(ProximityFeatureInput):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "locations": ("LOCATION",),
                "feature": ("FEATURE",),
                "transformation_type": (["translate", "scale"],),
                "transformation_value": ("FLOAT", {"default": 1.0}),
            }
        }

    RETURN_TYPES = ("LOCATION",)
    FUNCTION = "transform_locations"

    def transform_locations(self, locations, feature, transformation_type, transformation_value):
        transformed_locations = []
        for frame_index, location in enumerate(locations):
            feature_value = feature.get_value_at_frame(frame_index)
            if transformation_type == "translate":
                new_x = location.x + feature_value * transformation_value
                new_y = location.y + feature_value * transformation_value
                if location.z is not None:
                    new_z = location.z + feature_value * transformation_value
                    transformed_location = Location(new_x, new_y, new_z)
                else:
                    transformed_location = Location(new_x, new_y)
            elif transformation_type == "scale":
                new_x = location.x * feature_value * transformation_value
                new_y = location.y * feature_value * transformation_value
                if location.z is not None:
                    new_z = location.z * feature_value * transformation_value
                    transformed_location = Location(new_x, new_y, new_z)
                else:
                    transformed_location = Location(new_x, new_y)
            transformed_locations.append(transformed_location)

        return (transformed_locations,)



class ProximityVisualizer(RyanOnTheInside):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "anchor_locations": ("LOCATION",),
                "query_locations": ("LOCATION",),
                "proximity_feature": ("FEATURE",),
                "anchor_color": ("STRING", {"default": "(255,0,0)"}),
                "query_color": ("STRING", {"default": "(0,255,0)"}),
                "line_color": ("STRING", {"default": "(0,0,255)"}),
                "text_color": ("STRING", {"default": "(255,255,255)"}),
                "font_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_proximity"

    def visualize_proximity(self, video_frames, anchor_locations, query_locations, proximity_feature, 
                            anchor_color, query_color, line_color, text_color, font_scale):
        anchor_color = self.parse_color(anchor_color)
        query_color = self.parse_color(query_color)
        line_color = self.parse_color(line_color)
        text_color = self.parse_color(text_color)

        output_frames = []
        height, width = video_frames.shape[1:3]  # Extract height and width from video_frames

        # Calculate the frame diagonal from the video frames
        frame_diagonal = np.sqrt(width**2 + height**2)
        scale_factor = frame_diagonal / proximity_feature.frame_diagonal
        for frame_index in range(len(video_frames)):
            frame = video_frames[frame_index].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            anchor = anchor_locations[frame_index]
            query = query_locations[frame_index]

            # Calculate scaling factors based on the frame diagonal
            

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
                anchor_text = f"Anchor: ({int(closest_anchor[0] * scale_factor)}, {int(closest_anchor[1] * scale_factor)}"
                query_text = f"Query: ({int(closest_query[0] * scale_factor)}, {int(closest_query[1] * scale_factor)}"
                
                if anchor.z is not None and query.z is not None:
                    anchor_text += f", {closest_anchor[2]:.2f})"
                    query_text += f", {closest_query[2]:.2f})"
                else:
                    anchor_text += ") [2D]"
                    query_text += ") [2D]"

                cv2.putText(frame, anchor_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)
                cv2.putText(frame, query_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

            # Add proximity value to the frame
            proximity_value = proximity_feature.get_value_at_frame(frame_index)
            cv2.putText(frame, f"Proximity: {proximity_value:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)

        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)

    def parse_color(self, color_string):
        return tuple(map(int, color_string.strip("()").split(",")))