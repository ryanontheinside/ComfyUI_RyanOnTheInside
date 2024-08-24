def string_to_rgb(color_string):
    if isinstance(color_string, tuple):
        return color_string
    color_values = color_string.strip('()').split(',')
    return tuple(int(value.strip()) / 255.0 for value in color_values)