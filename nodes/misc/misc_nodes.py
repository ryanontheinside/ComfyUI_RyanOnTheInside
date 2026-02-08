import json


class ROTIDisplayAny:
    """Debug node that displays any value as text."""

    CATEGORY = "RyanOnTheInside/Misc"
    FUNCTION = "display"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("*",),
            },
        }

    def display(self, source=None):
        if source is None:
            value = "None"
        elif isinstance(source, (str, int, float, bool)):
            value = str(source)
        else:
            try:
                value = json.dumps(source, indent=2, default=str)
            except Exception:
                try:
                    value = str(source)
                except Exception:
                    value = "Could not serialize value"

        return {"ui": {"text": (value,)}}
