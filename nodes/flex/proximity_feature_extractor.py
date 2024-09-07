from .proximity_feature import Location,   ProximityFeature
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
                "distance_metric": (["euclidean", "manhattan", "chebyshev"],),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    RETURN_NAMES = ("proximity_feature", "feature_pipe")
    FUNCTION = "create_feature"
    CATEGORY = "RyanOnTheInside/FlexFeatures"

    def create_feature(self, video_frames, frame_rate, anchor_locations, query_locations, distance_metric):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        
        proximity_feature = ProximityFeature(
            name="proximity_feature",
            anchor_locations=anchor_locations,
            query_locations=query_locations,
            frame_rate=frame_rate,
            frame_count=len(video_frames),
            distance_metric=distance_metric
        )

        self.start_progress(len(video_frames), desc="Calculating proximity")
        
        proximity_feature.extract()
        
        for _ in range(len(video_frames)):
            self.update_progress()

        self.end_progress()

        return (proximity_feature, feature_pipe)

class LocationFromMask(RyanOnTheInside):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "method": (["mask_center", "mask_boundary", "mask_top_left", "mask_bottom_right"],),
            },
            "optional": {
                "depth_maps": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LOCATION",)
    FUNCTION = "generate_locations"
    CATEGORY = "RyanOnTheInside/Proximity"

    def generate_locations(self, masks, method, depth_maps=None):
        num_frames = masks.shape[0]
        locations = []

        # self.start_progress(num_frames, desc="Generating locations")

        for i in range(num_frames):
            mask = masks[i].cpu().numpy()
            y, x = np.where(mask > 0.5)

            if len(x) == 0 or len(y) == 0:
                location = Location(mask.shape[1] / 2, mask.shape[0] / 2)
            else:
                if method == "mask_center":
                    center_x, center_y = np.mean(x), np.mean(y)
                    location = Location(center_x, center_y)
                elif method == "mask_boundary":
                    contour = np.column_stack((x, y))
                    epsilon = 0.01 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    location = Location(approx[:, 0, 0].tolist(), approx[:, 0, 1].tolist())
                elif method == "mask_top_left":
                    location = Location(np.min(x), np.min(y))
                elif method == "mask_bottom_right":
                    location = Location(np.max(x), np.max(y))
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if depth_maps is not None:
                    for point in location:
                        point.z = depth_maps[i, int(point.y), int(point.x)].mean().item()

            locations.append(location)
            # self.update_progress()

        # self.end_progress()

        return (locations,)