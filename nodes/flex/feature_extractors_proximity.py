from .features_proximity import Location,   ProximityFeature
from .feature_extractors import FirstFeature
from ... import RyanOnTheInside
from .feature_pipe import FeaturePipe
import numpy as np
import cv2


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



