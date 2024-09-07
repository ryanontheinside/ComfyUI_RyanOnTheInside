from .features import  BaseFeature
import torch
import numpy as np

import numpy as np
from .features import BaseFeature

class ProximityFeature(BaseFeature):
    def __init__(self, name, anchor_locations, query_locations, frame_rate, frame_count, distance_metric='euclidean'):
        super().__init__(name, "proximity", frame_rate, frame_count)
        self.anchor_locations = anchor_locations
        self.query_locations = query_locations
        self.distance_metric = distance_metric
        self.closest_distances = None
        self.global_min = None
        self.global_max = None
        self.proximity_values = None

    def extract(self):
        self.closest_distances = []
        all_distances = []

        for anchor, query in zip(self.anchor_locations, self.query_locations):
            frame_distances = []
            for a in anchor:
                for q in query:
                    if self.distance_metric == 'euclidean':
                        distance = np.sqrt((a.x - q.x)**2 + (a.y - q.y)**2 + (a.z - q.z)**2)
                    elif self.distance_metric == 'manhattan':
                        distance = abs(a.x - q.x) + abs(a.y - q.y) + abs(a.z - q.z)
                    elif self.distance_metric == 'chebyshev':
                        distance = max(abs(a.x - q.x), abs(a.y - q.y), abs(a.z - q.z))
                    else:
                        raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
                    frame_distances.append(distance)
            
            closest_distance = min(frame_distances)
            self.closest_distances.append(closest_distance)
            all_distances.extend(frame_distances)

        self.closest_distances = np.array(self.closest_distances)
        self.global_min = np.min(all_distances)
        self.global_max = np.max(all_distances)

        return self.normalize()

    def normalize(self):
        if self.global_max > self.global_min:
            # Normalize and invert the values
            self.proximity_values = 1 - (self.closest_distances - self.global_min) / (self.global_max - self.global_min)
        else:
            self.proximity_values = np.ones_like(self.closest_distances)
        return self

    def get_value_at_frame(self, frame_index):
        return self.proximity_values[frame_index]

class Location:
    def __init__(self, x, y, z=0.5):
        if isinstance(x, (list, tuple, np.ndarray)):
            self.points = []
            for i in range(len(x)):
                xi = x[i]
                yi = y[i] if isinstance(y, (list, tuple, np.ndarray)) else y
                zi = z[i] if isinstance(z, (list, tuple, np.ndarray)) else z
                self.points.append(Location(xi, yi, zi))
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.points = [self]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    def __iter__(self):
        return iter(self.points)

    def __repr__(self):
        if len(self) == 1:
            return f"Location(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"
        else:
            return f"Location(points={len(self)})"

    @classmethod
    def from_tensor(cls, tensor):
        if tensor.dim() == 1:
            return cls(tensor[0].item(), tensor[1].item(), tensor[2].item() if len(tensor) > 2 else 0.5)
        elif tensor.dim() == 2:
            return cls(tensor[:, 0].tolist(), tensor[:, 1].tolist(), 
                       tensor[:, 2].tolist() if tensor.shape[1] > 2 else [0.5] * tensor.shape[0])
        else:
            raise ValueError("Tensor must have 1 or 2 dimensions")

    def to_tensor(self):
        return torch.tensor([[p.x, p.y, p.z] for p in self.points])