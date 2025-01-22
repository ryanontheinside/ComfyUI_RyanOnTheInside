from .audio_nodes import AudioNodeBase
from .audio_utils import (
    pitch_shift,
    fade_audio,
    apply_gain,
    time_stretch,
)
from ...tooltips import apply_tooltips


class AudioEffect(AudioNodeBase):
    def __init__(self):
        super().__init__()

    CATEGORY = "RyanOnTheInside/Audio/Effects"

@apply_tooltips
class AudioPitchShift(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "n_steps": ("INT", {"default": 0, "min": -12, "max": 12, "step": 1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pitch_shift_audio"

    def pitch_shift_audio(self, audio, n_steps):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        shifted_waveform = pitch_shift(waveform, sample_rate, n_steps)
        return ({"waveform": shifted_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class AudioFade(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "fade_in_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "fade_out_duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "shape": (["linear", "exponential", "logarithmic", "quarter_sine", "half_sine"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "fade_audio_node"

    def fade_audio_node(self, audio, fade_in_duration, fade_out_duration, shape):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        faded_waveform = fade_audio(waveform, sample_rate, fade_in_duration, fade_out_duration, shape)
        return ({"waveform": faded_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class AudioGain(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_gain_node"

    def apply_gain_node(self, audio, gain_db):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        amplified_waveform = apply_gain(waveform, gain_db)
        return ({"waveform": amplified_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class AudioTimeStretch(AudioEffect):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "rate": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "time_stretch_node"

    def time_stretch_node(self, audio, rate):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        stretched_waveform = time_stretch(waveform, rate)
        return ({"waveform": stretched_waveform, "sample_rate": sample_rate},)