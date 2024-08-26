from ... import RyanOnTheInside

class FlexExternalModulator(RyanOnTheInside):
    CATEGORY = "RyanOnTheInside/FlexExternalMod"

class FeatureToWeightsStrategy(FlexExternalModulator):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature": ("FEATURE",),
            }
        }

    RETURN_TYPES = ("WEIGHTS_STRATEGY",)
    RETURN_NAMES = ("WEIGHTS_STRATEGY",)
    FUNCTION = "convert"

    def convert(self, feature):
        frames = feature.frame_count
        values = [feature.get_value_at_frame(i) for i in range(frames)]
        
        weights_str = ", ".join(map(lambda x: f"{x:.8f}", values))

        weights_strategy = {
            "weights": weights_str,
            "timing": "custom",
            "frames": frames,
            "start_frame": 0,
            "end_frame": frames,
            "add_starting_frames": 0,
            "add_ending_frames": 0,
            "method": "full batch",
            "frame_count": frames,
        }


        return (weights_strategy,)
