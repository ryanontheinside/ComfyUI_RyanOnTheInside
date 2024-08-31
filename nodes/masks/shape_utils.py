import numpy as np
from matplotlib.path import Path

def create_circle(center, size, params):
    return lambda x, y: ((x - center[0])**2 + (y - center[1])**2 <= size**2)

def create_ellipse(center, size, params):
    aspect_ratio = params.get('aspect_ratio', 0.5)
    return lambda x, y: ((x - center[0])**2 / size**2 + (y - center[1])**2 / (size * aspect_ratio)**2 <= 1)

def create_rectangle(center, size, params):
    aspect_ratio = params.get('aspect_ratio', 1)
    return lambda x, y: (abs(x - center[0]) <= size) & (abs(y - center[1]) <= size * aspect_ratio)

def create_polygon(center, size, params):
    sides = params.get('sides', 3)
    rotation = params.get('rotation', 0)
    angles = np.linspace(0, 2*np.pi, sides, endpoint=False) + rotation
    polygon_x = center[0] + size * np.cos(angles)
    polygon_y = center[1] + size * np.sin(angles)
    path = Path(np.column_stack([polygon_x, polygon_y]))
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_star(center, size, params):
    num_points = params.get('num_points', 5)
    inner_radius = size * params.get('inner_radius_ratio', 0.5)
    rotation = params.get('rotation', 0)
    angles = np.linspace(0, 2*np.pi, 2*num_points, endpoint=False) + rotation
    radii = np.array([size, inner_radius] * num_points)
    star_x = center[0] + radii * np.cos(angles)
    star_y = center[1] + radii * np.sin(angles)
    path = Path(np.column_stack([star_x, star_y]))
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_cross(center, size, params):
    thickness = params.get('thickness', 0.2)
    return lambda x, y: ((abs(x - center[0]) <= size * thickness) | (abs(y - center[1]) <= size * thickness)) & \
                        ((abs(x - center[0]) <= size) & (abs(y - center[1]) <= size))

def create_heart(center, size, params):
    def heart_curve(t):
        x = 16 * np.sin(t)**3
        y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
        return x, -y
    t = np.linspace(0, 2*np.pi, 100)
    heart_x, heart_y = heart_curve(t)
    heart_x = center[0] + size * heart_x / np.max(np.abs(heart_x))
    heart_y = center[1] + size * heart_y / np.max(np.abs(heart_y))
    path = Path(np.column_stack([heart_x, heart_y]))
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_crescent(center, size, params):
    inner_radius = size * params.get('inner_radius_ratio', 0.7)
    offset = size * params.get('offset_ratio', 0.3)
    return lambda x, y: ((x - center[0])**2 + (y - center[1])**2 <= size**2) & \
                        ((x - (center[0] + offset))**2 + (y - center[1])**2 > inner_radius**2)

def create_arrow(center, size, params):
    width = params.get('width', 0.5) * size
    head_width = params.get('head_width', 1.5) * width
    head_length = params.get('head_length', 0.3) * size
    rotation = params.get('rotation', 0)
    points = np.array([
        [0, -width/2], [size-head_length, -width/2], [size-head_length, -head_width/2],
        [size, 0], [size-head_length, head_width/2], [size-head_length, width/2],
        [0, width/2]
    ])
    points = points - [size/2, 0]  # Center the arrow
    cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_points = np.dot(points, rotation_matrix.T)
    translated_points = rotated_points + center
    path = Path(translated_points)
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_plus(center, size, params):
    thickness = params.get('thickness', 0.2)
    return lambda x, y: (((abs(x - center[0]) <= size * thickness) & (abs(y - center[1]) <= size)) |
                         ((abs(x - center[0]) <= size) & (abs(y - center[1]) <= size * thickness)))

def create_donut(center, size, params):
    inner_radius = size * params.get('inner_radius_ratio', 0.5)
    return lambda x, y: ((x - center[0])**2 + (y - center[1])**2 <= size**2) & \
                        ((x - center[0])**2 + (y - center[1])**2 > inner_radius**2)

def create_spiral(center, size, params):
    turns = params.get('turns', 3)
    thickness = params.get('thickness', 0.1)
    def spiral(t):
        x = size * t * np.cos(2*np.pi*turns*t) / turns
        y = size * t * np.sin(2*np.pi*turns*t) / turns
        return np.column_stack([x, y])
    t = np.linspace(0, 1, 1000)
    spiral_points = spiral(t) + center
    path = Path(spiral_points, [Path.MOVETO] + [Path.LINETO] * (len(spiral_points) - 1))
    return lambda x, y: path.contains_points(np.column_stack([x, y])) | \
                        (((x - center[0])**2 + (y - center[1])**2 <= (size*thickness)**2))

def create_gear(center, size, params):
    num_teeth = params.get('num_teeth', 20)
    tooth_depth = params.get('tooth_depth', 0.1)
    rotation = params.get('rotation', 0)
    angles = np.linspace(0, 2*np.pi, num_teeth*2, endpoint=False) + rotation
    radii = np.array([size, size * (1 - tooth_depth)] * num_teeth)
    gear_x = center[0] + radii * np.cos(angles)
    gear_y = center[1] + radii * np.sin(angles)
    path = Path(np.column_stack([gear_x, gear_y]))
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_flower(center, size, params):
    num_petals = params.get('num_petals', 5)
    petal_width = params.get('petal_width', 0.3)
    rotation = params.get('rotation', 0)
    t = np.linspace(0, 2*np.pi, 1000)
    r = size * np.abs(np.sin(num_petals/2 * t)) ** (1/petal_width)
    flower_x = center[0] + r * np.cos(t + rotation)
    flower_y = center[1] + r * np.sin(t + rotation)
    path = Path(np.column_stack([flower_x, flower_y]))
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_trapezoid(center, size, params):
    top_width = params.get('top_width', 0.5)
    rotation = params.get('rotation', 0)
    points = np.array([[-size, -size], [size, -size], [size*top_width, size], [-size*top_width, size]])
    cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_points = np.dot(points, rotation_matrix.T)
    translated_points = rotated_points + center
    path = Path(translated_points)
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_parallelogram(center, size, params):
    skew = params.get('skew', 0.5)
    rotation = params.get('rotation', 0)
    points = np.array([[-size+size*skew, -size], [size+size*skew, -size], [size, size], [-size, size]])
    cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_points = np.dot(points, rotation_matrix.T)
    translated_points = rotated_points + center
    path = Path(translated_points)
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_rhombus(center, size, params):
    aspect_ratio = params.get('aspect_ratio', 0.5)
    rotation = params.get('rotation', 0)
    points = np.array([[0, -size], [size*aspect_ratio, 0], [0, size], [-size*aspect_ratio, 0]])
    cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_points = np.dot(points, rotation_matrix.T)
    translated_points = rotated_points + center
    path = Path(translated_points)
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_semicircle(center, size, params):
    rotation = params.get('rotation', 0)
    t = np.linspace(0, np.pi, 100)
    semicircle_x = center[0] + size * np.cos(t + rotation)
    semicircle_y = center[1] + size * np.sin(t + rotation)
    points = np.column_stack([semicircle_x, semicircle_y])
    points = np.vstack([points, [center[0], center[1]]])
    path = Path(points)
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_quarter_circle(center, size, params):
    rotation = params.get('rotation', 0)
    t = np.linspace(0, np.pi/2, 50)
    quarter_circle_x = center[0] + size * np.cos(t + rotation)
    quarter_circle_y = center[1] + size * np.sin(t + rotation)
    points = np.column_stack([quarter_circle_x, quarter_circle_y])
    points = np.vstack([points, [center[0], center[1]]])
    path = Path(points)
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_pie_slice(center, size, params):
    angle = params.get('angle', np.pi/4)
    rotation = params.get('rotation', 0)
    t = np.linspace(0, angle, 50)
    pie_x = center[0] + size * np.cos(t + rotation)
    pie_y = center[1] + size * np.sin(t + rotation)
    points = np.column_stack([pie_x, pie_y])
    points = np.vstack([[center[0], center[1]], points])
    path = Path(points)
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_teardrop(center, size, params):
    rotation = params.get('rotation', 0)
    t = np.linspace(0, 2*np.pi, 100)
    r = size * (1 - np.cos(t)) / 2
    teardrop_x = center[0] + r * np.cos(t + rotation)
    teardrop_y = center[1] + r * np.sin(t + rotation)
    path = Path(np.column_stack([teardrop_x, teardrop_y]))
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_starburst(center, size, params):
    num_points = params.get('num_points', 8)
    inner_radius = size * params.get('inner_radius_ratio', 0.5)
    rotation = params.get('rotation', 0)
    angles = np.linspace(0, 2*np.pi, num_points*2, endpoint=False) + rotation
    radii = np.array([size, inner_radius] * num_points)
    starburst_x = center[0] + radii * np.cos(angles)
    starburst_y = center[1] + radii * np.sin(angles)
    path = Path(np.column_stack([starburst_x, starburst_y]))
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

def create_cloud(center, size, params):
    num_bulbs = params.get('num_bulbs', 5)
    bulb_size = size * 0.5
    angles = np.linspace(0, 2*np.pi, num_bulbs, endpoint=False)
    cloud_x = center[0] + size * 0.5 * np.cos(angles)
    cloud_y = center[1] + size * 0.25 * np.sin(angles)
    
    def cloud_shape(x, y):
        mask = np.zeros_like(x, dtype=bool)
        for cx, cy in zip(cloud_x, cloud_y):
            mask |= ((x - cx)**2 + (y - cy)**2 <= bulb_size**2)
        return mask
    
    return cloud_shape

def create_raindrop(center, size, params):
    rotation = params.get('rotation', 0)
    t = np.linspace(0, 2*np.pi, 100)
    r = size * (1 - 0.5*np.sin(t/2))
    raindrop_x = center[0] + r * np.cos(t + rotation)
    raindrop_y = center[1] + r * np.sin(t + rotation)
    path = Path(np.column_stack([raindrop_x, raindrop_y]))
    return lambda x, y: path.contains_points(np.column_stack([x, y]))

# Dictionary mapping shape names to their creation functions
SHAPE_FUNCTIONS = {
    "circle": create_circle,
    "ellipse": create_ellipse,
    "rectangle": create_rectangle,
    "triangle": lambda center, size, params: create_polygon(center, size, {"sides": 3, **params}),
    "square": lambda center, size, params: create_rectangle(center, size, {"aspect_ratio": 1, **params}),
    "pentagon": lambda center, size, params: create_polygon(center, size, {"sides": 5, **params}),
    "hexagon": lambda center, size, params: create_polygon(center, size, {"sides": 6, **params}),
    "octagon": lambda center, size, params: create_polygon(center, size, {"sides": 8, **params}),
    "star": create_star,
    "cross": create_cross,
    "heart": create_heart,
    "crescent": create_crescent,
    "arrow": create_arrow,
    "plus": create_plus,
    "donut": create_donut,
    "spiral": create_spiral,
    "gear": create_gear,
    "flower": create_flower,
    "trapezoid": create_trapezoid,
    "parallelogram": create_parallelogram,
    "rhombus": create_rhombus,
    "semicircle": create_semicircle,
    "quarter_circle": create_quarter_circle,
    "pie_slice": create_pie_slice,
    "teardrop": create_teardrop,
    "starburst": create_starburst,
    "cloud": create_cloud,
    "raindrop": create_raindrop,
}

def create_shape_mask(shape, center, shape_type, size, params=None):
    height, width = shape
    y, x = np.ogrid[:height, :width]
    params = params or {}

    if shape_type in SHAPE_FUNCTIONS:
        mask_func = SHAPE_FUNCTIONS[shape_type](center, size, params)
        # Ensure x and y have the same shape
        x_flat, y_flat = np.broadcast_arrays(x, y)
        points = np.column_stack([x_flat.ravel(), y_flat.ravel()])
        mask = mask_func(points[:, 0], points[:, 1]).reshape(shape)
        return mask.astype(np.float32)
    else:
        return np.zeros(shape, dtype=np.float32)

# Function to get all available shape types
def get_available_shapes():
    return list(SHAPE_FUNCTIONS.keys()) + ["random"]
