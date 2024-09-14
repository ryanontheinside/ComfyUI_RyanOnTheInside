from .feature_extractors import FeatureExtractorBase
from .features_audio import AudioFeature, PitchFeature, PitchRange, BaseFeature
from ... import RyanOnTheInside

class AudioFeatureExtractor(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return AudioFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "audio": ("AUDIO",),
                "feature_pipe": ("FEATURE_PIPE",),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "extract_feature"
    

    def extract_feature(self, audio, feature_pipe, extraction_method):
        feature = AudioFeature(extraction_method, audio, feature_pipe.frame_count, feature_pipe.frame_rate, extraction_method)
        feature.extract()
        return (feature, feature_pipe)

class PitchFeatureExtractor(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return  PitchFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {            
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "audio": ("AUDIO", ),
                "feature_pipe": ("FEATURE_PIPE", ),
                "window_size": ("INT", {"default": 0, "min": 0}),
                "pitch_tolerance_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max":1.0, "step": 0.01}),
            },
            "optional": {
                "pitch_range_collections": ("PITCH_RANGE_COLLECTION", ),
            },
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "extract_feature"

    CATEGORY = "RyanOnTheInside/FlexFeatures"

    def extract_feature(self, audio, feature_pipe, extraction_method, window_size, pitch_tolerance_percent, pitch_range_collections=None):
        if pitch_range_collections is None:
            pitch_range_collections = []
        feature = PitchFeature(
            feature_name="PitchFeature",
            audio=audio,
            frame_count=feature_pipe.frame_count,
            frame_rate=feature_pipe.frame_rate,
            pitch_range_collections=pitch_range_collections,
            feature_type=extraction_method,
            window_size=window_size,
            pitch_tolerance_percent=pitch_tolerance_percent,
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
        return {
            "required": {
                "chord_only": ("BOOLEAN", {"default": False}),
                "pitch_tolerance_percent": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "notes": ("STRING", {"multiline": False}),
            },
            "optional": {
                "previous_range_collection": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("PITCH_RANGE_COLLECTION",)
    FUNCTION = "create_note_pitch_ranges"

    CATEGORY = "RyanOnTheInside/FlexFeatures"

    def create_note_pitch_ranges(self, chord_only, notes, pitch_tolerance_percent, previous_range_collection=None):
        if not notes:
            raise ValueError("At least one note must be selected.")

        # Parse the 'notes' string into a list of MIDI note numbers
        selected_notes = [int(note.strip()) for note in notes.split(',') if note.strip().isdigit()]

        if not selected_notes:
            raise ValueError("No valid notes found in the 'notes' field.")

        pitch_ranges = []
        for midi_note in selected_notes:
            frequency = self._midi_to_frequency(midi_note)
            tolerance = PitchFeature.calculate_tolerance(frequency, pitch_tolerance_percent)
            min_pitch = frequency - tolerance
            max_pitch = frequency + tolerance
            pitch_range = PitchRange(min_pitch, max_pitch)
            pitch_ranges.append(pitch_range)

        # Create a collection with the 'chord_only' attribute
        pitch_range_collection = {
            "pitch_ranges": pitch_ranges,
            "chord_only": chord_only,
        }

        # Combine with previous collections if provided
        if previous_range_collection is None:
            collections = [pitch_range_collection]
        else:
            collections = previous_range_collection + [pitch_range_collection]

        return (collections,)

    def _midi_to_frequency(self, midi_note):
        import librosa
        return librosa.midi_to_hz(midi_note)

    # The _calculate_tolerance method has been removed from here