from .feature_extractors import FeatureExtractorBase
from .audio_feature import AudioFeature, PitchFeature, PitchRange
from ... import RyanOnTheInside
class AudioFeatureExtractor(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "feature_pipe": ("FEATURE_PIPE",),
                "feature_type": (["amplitude_envelope",  "spectral_centroid", "onset_detection", "chroma_features"],),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "extract_feature"
    

    def extract_feature(self, audio, feature_pipe, feature_type):
        feature = AudioFeature(feature_type, audio, feature_pipe.frame_count, feature_pipe.frame_rate, feature_type)
        feature.extract()
        return (feature, feature_pipe)

class PitchFeatureExtractor(FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "feature_pipe": ("FEATURE_PIPE", ),
                "feature_type": (
                    [
                        "pitch",
                        "pitch_filtered",
                        "pitch_direction",
                        "vibrato_signal",
                        "vibrato_intensity",
                    ],
                ),
                "window_size": ("INT", {"default": 0, "min": 0}),
                "pitch_tolerance": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
            },
            "optional": {
                "pitch_range_collections": ("PITCH_RANGE_COLLECTION", ),
            },
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "extract_feature"

    CATEGORY = "RyanOnTheInside/FlexFeatures"

    def extract_feature(self, audio, feature_pipe, feature_type, window_size, pitch_tolerance, pitch_range_collections=None):
        if pitch_range_collections is None:
            pitch_range_collections = []
        feature = PitchFeature(
            feature_name="PitchFeature",
            audio=audio,
            frame_count=feature_pipe.frame_count,
            frame_rate=feature_pipe.frame_rate,
            pitch_range_collections=pitch_range_collections,
            feature_type=feature_type,
            window_size=window_size,
            pitch_tolerance=pitch_tolerance,
        )
        feature.extract()
        return (feature, feature_pipe)

class PitchAbstraction(RyanOnTheInside):
    CATEGORY="RyanOnTheInside/FlexFeatures/Audio"

class PitchRangeNode(PitchAbstraction):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_pitch": ("FLOAT", {"default": 80.0, "min": 20.0, "max": 2000.0, "step": 1.0}),
                "max_pitch": ("FLOAT", {"default": 400.0, "min": 20.0, "max": 2000.0, "step": 1.0}),
            },
            "optional": {
                "previous_range_collection": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("PITCH_RANGE_COLLECTION",)
    FUNCTION = "create_pitch_range"

    CATEGORY = "RyanOnTheInside/FlexFeatures"

    def create_pitch_range(self, min_pitch, max_pitch, previous_range_collection=None):
        pitch_range = PitchRange(min_pitch, max_pitch)
        pitch_range_collection = {
            "pitch_ranges": [pitch_range],
            "chord_only": False,
        }
        if previous_range_collection is None:
            collections = [pitch_range_collection]
        else:
            collections = previous_range_collection + [pitch_range_collection]
        return (collections,)
    
class PitchRangePresetNode(PitchAbstraction):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    [
                        "Bass",
                        "Baritone",
                        "Tenor",
                        "Alto",
                        "Mezzo-soprano",
                        "Soprano",
                        "Contralto",
                    ],
                )
            },
            "optional": {
                "previous_range_collection": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("PITCH_RANGE_COLLECTION",)
    FUNCTION = "create_pitch_range_preset"

    CATEGORY = "RyanOnTheInside/FlexFeatures"

    def create_pitch_range_preset(self, preset, previous_range_collection=None):
        presets = {
            "Bass": (82.41, 196.00),            # E2 - G3
            "Baritone": (98.00, 247.94),        # G2 - B3
            "Tenor": (130.81, 349.23),          # C3 - F4
            "Contralto": (130.81, 349.23),      # C3 - F4
            "Alto": (174.61, 440.00),           # F3 - A4
            "Mezzo-soprano": (196.00, 523.25),  # G3 - C5
            "Soprano": (261.63, 1046.50),       # C4 - C6
        }

        min_pitch, max_pitch = presets.get(preset, (20.0, 2000.0))
        pitch_range = PitchRange(min_pitch, max_pitch)
        pitch_range_collection = {
            "pitch_ranges": [pitch_range],
            "chord_only": False,
        }
        if previous_range_collection is None:
            collections = [pitch_range_collection]
        else:
            collections = previous_range_collection + [pitch_range_collection]
        return (collections,)
    
class PitchRangeByNoteNode(PitchAbstraction):
    @classmethod
    def INPUT_TYPES(cls):
        notes = [
            "C", "C#", "D", "D#", "E", "F", "F#",
            "G", "G#", "A", "A#", "B"
        ]
        octaves = range(0, 9)  # MIDI supports octaves from 0 to 8
        note_options = [f"{note}{octave}" for octave in octaves for note in notes]
        return {
            "required": {
                "chord_only": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                **{note: ("BOOLEAN", {"default": False}) for note in note_options},
                "previous_range_collection": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("PITCH_RANGE_COLLECTION",)
    FUNCTION = "create_note_pitch_ranges"

    CATEGORY = "RyanOnTheInside/FlexFeatures"

    def create_note_pitch_ranges(self, chord_only, previous_range_collection=None, **note_selection):
        selected_notes = [note for note, selected in note_selection.items() if selected]
        if not selected_notes:
            raise ValueError("At least one note must be selected.")

        pitch_ranges = []
        for note in selected_notes:
            frequency = self._note_to_frequency(note)
            # Define a small range around the note's frequency
            min_pitch = frequency - 1.0  # 1 Hz below
            max_pitch = frequency + 1.0  # 1 Hz above
            pitch_range = PitchRange(min_pitch, max_pitch)
            pitch_ranges.append(pitch_range)

        # Create a collection with the 'chord_only' attribute
        pitch_range_collection = {
            "pitch_ranges": pitch_ranges,
            "chord_only": chord_only,
        }

        if previous_range_collection is None:
            collections = [pitch_range_collection]
        else:
            collections = previous_range_collection + [pitch_range_collection]

        return (collections,)

    def _note_to_frequency(self, note):
        import librosa
        return librosa.note_to_hz(note)