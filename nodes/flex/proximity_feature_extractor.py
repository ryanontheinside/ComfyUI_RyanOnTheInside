from .proximity_feature import Location,   ProximityFeature
from .feature_extractors import FirstFeature
from ... import RyanOnTheInside
from .feature_pipe import FeaturePipe
import numpy as np
import cv2
import  torch

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
                "anchor_boundary_color": ("STRING", {"default": "(255,128,0)"}),
                "query_boundary_color": ("STRING", {"default": "(0,255,128)"}),
                "point_size": ("INT", {"default": 5, "min": 1, "max": 20}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
                "boundary_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_proximity"
    CATEGORY = "RyanOnTheInside/Visualization"

    def parse_color(self, color_string):
        return tuple(map(int, color_string.strip("()").split(",")))

    def draw_boundary(self, frame, locations, color, thickness):
        print(f"Number of points in locations: {len(locations)}")  # Debugging statement
        if len(locations) < 3:
            return  # Not enough points to form a boundary
        from scipy.spatial import ConvexHull

        points = np.array([(loc.x, loc.y) for loc in locations])
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        hull_points = np.round(hull_points).astype(int)
        cv2.polylines(frame, [hull_points], True, color, thickness)

    def visualize_proximity(self, video_frames, anchor_locations, query_locations, proximity_feature, 
                            anchor_color, query_color, line_color, anchor_boundary_color, query_boundary_color, 
                            point_size, line_thickness, boundary_thickness):
        anchor_color = self.parse_color(anchor_color)
        query_color = self.parse_color(query_color)
        line_color = self.parse_color(line_color)
        anchor_boundary_color = self.parse_color(anchor_boundary_color)
        query_boundary_color = self.parse_color(query_boundary_color)

        output_frames = []

        for frame_index in range(len(video_frames)):
            frame = video_frames[frame_index].cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            anchor = anchor_locations[frame_index]
            query = query_locations[frame_index]

            # Draw boundaries formed by the location points
            self.draw_boundary(frame, anchor, anchor_boundary_color, boundary_thickness)
            self.draw_boundary(frame, query, query_boundary_color, boundary_thickness)

            # Draw anchor points
            for point in anchor:
                cv2.circle(frame, (int(point.x), int(point.y)), point_size, anchor_color, -1)

            # Draw query points
            for point in query:
                cv2.circle(frame, (int(point.x), int(point.y)), point_size, query_color, -1)

            # Find and draw the closest pair of points
            closest_anchor = None
            closest_query = None
            min_distance = float('inf')

            for a in anchor:
                for q in query:
                    distance = np.sqrt((a.x - q.x)**2 + (a.y - q.y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_anchor = a
                        closest_query = q

            if closest_anchor and closest_query:
                cv2.line(frame, (int(closest_anchor.x), int(closest_anchor.y)),
                         (int(closest_query.x), int(closest_query.y)), line_color, line_thickness)

            # Add proximity value to the frame
            proximity_value = proximity_feature.get_value_at_frame(frame_index)
            cv2.putText(frame, f"Proximity: {proximity_value:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frames.append(frame)

        output_tensor = torch.from_numpy(np.stack(output_frames)).float() / 255.0
        return (output_tensor,)