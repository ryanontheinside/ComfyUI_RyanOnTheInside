def string_to_rgb(color_string):
    if isinstance(color_string, tuple):
        return color_string
    color_values = color_string.strip('()').split(',')
    return tuple(int(value.strip()) / 255.0 for value in color_values)


def apply_easing(t, start, end, easing):
    if easing == "linear":
        return start + t * (end - start)
    elif easing == "ease_in_quad":
        return start + (t**2) * (end - start)
    elif easing == "ease_out_quad":
        return start + (1 - (1-t)**2) * (end - start)
    elif easing == "ease_in_out_quad":
        return start + (2*t**2 if t < 0.5 else 1-2*(1-t)**2) * (end - start)
    elif easing == "ease_in_cubic":
        return start + (t**3) * (end - start)
    elif easing == "ease_out_cubic":
        return start + (1 - (1-t)**3) * (end - start)
    elif easing == "ease_in_out_cubic":
        return start + (4*t**3 if t < 0.5 else 1-4*(1-t)**3) * (end - start)
    elif easing == "ease_in_quart":
        return start + (t**4) * (end - start)
    elif easing == "ease_out_quart":
        return start + (1 - (1-t)**4) * (end - start)
    elif easing == "ease_in_out_quart":
        return start + (8*t**4 if t < 0.5 else 1-8*(1-t)**4) * (end - start)