from .features import  BaseFeature
import torch
import numpy as np

import numpy as np
from .features import BaseFeature

class ProximityFeature(BaseFeature):
    def __init__(self, name, anchor_locations, query_locations, frame_rate, frame_count, frame_dimensions, width, height, normalization_method='frame'):
        super().__init__(name, "proximity", frame_rate, frame_count, width, height)
        self.anchor_locations = anchor_locations
        self.query_locations = query_locations
        self.frame_diagonal = np.sqrt(frame_dimensions[0]**2 + frame_dimensions[1]**2)
        self.proximity_values = None
        self.normalization_method = normalization_method

    @classmethod
    def get_extraction_methods(self):
        return ["normalization_method"]
    
    def extract(self):
        from scipy.spatial.distance import cdist
        
        proximities = []
        for anchor, query in zip(self.anchor_locations, self.query_locations):
            if len(anchor) == 0 or len(query) == 0:
                proximities.append(self.frame_diagonal if self.normalization_method == 'frame' else float('inf'))
            else:
                # Ensure points are treated as floats
                anchor_points = anchor.points.astype(float)
                query_points = query.points.astype(float)
                distances = cdist(anchor_points, query_points)
                min_distance = np.min(distances)
                proximities.append(min_distance)
        
        proximities = np.array(proximities, dtype=float)
        
        if self.normalization_method == 'frame':
            self.proximity_values = 1 - np.clip(proximities / self.frame_diagonal, 0, 1)
        elif self.normalization_method == 'minmax':
            min_proximity = np.min(proximities)
            max_proximity = np.max(proximities)
            if max_proximity > min_proximity:
                self.proximity_values = 1 - (proximities - min_proximity) / (max_proximity - min_proximity)
            else:
                self.proximity_values = np.ones_like(proximities)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        self.smooth_proximities()
        return self

    def smooth_proximities(self, window_size=5):
        kernel = np.ones(window_size) / window_size
        self.proximity_values = np.convolve(self.proximity_values, kernel, mode='same')

    def get_value_at_frame(self, frame_index):
        return self.proximity_values[frame_index]

class Location:
    def __init__(self, x, y, z=None):
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        
        if z is not None:
            z = np.asarray(z, dtype=float).reshape(-1)
            if abs(len(z) - len(x)) <= 1:
                z = z[:len(x)] if len(z) > len(x) else np.pad(z, (0, len(x) - len(z)), 'constant', constant_values=0.0)
            elif len(z) != len(x):
                raise ValueError(f"Mismatch in dimensions: x,y have length {len(x)}, but z has length {len(z)}")
            self.points = np.column_stack((x, y, z))
        else:
            self.points = np.column_stack((x, y))
        

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    @property
    def x(self):
        return self.points[:, 0]

    @property
    def y(self):
        return self.points[:, 1]

    @property
    def z(self):
        return self.points[:, 2] if self.points.shape[1] > 2 else None

    @classmethod
    def from_tensor(cls, tensor):
        return cls(tensor[:, 0].astype(float), tensor[:, 1].astype(float), tensor[:, 2].astype(float) if tensor.shape[1] > 2 else None)

    def to_tensor(self):
        return torch.from_numpy(self.points).float()