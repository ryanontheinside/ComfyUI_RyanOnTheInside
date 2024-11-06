from .feature_extractors import FeatureExtractorBase, FirstFeature
from .feature_pipe import FeaturePipe
from .features_audio import AudioFeature, PitchFeature, PitchRange, BaseFeature, RhythmFeature
from ... import RyanOnTheInside
from ..audio.audio_nodes import AudioNodeBase

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

#todo: create base class in prep for version 2
class AudioFeatureExtractorFirst(FeatureExtractorBase):
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
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 240.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE", "INT")
    RETURN_NAMES = ("feature", "feature_pipe", "frame_count")
    FUNCTION = "extract_feature"
    CATEGORY = "RyanOnTheInside/FlexFeatures/Audio"

    def extract_feature(self, audio, width, height, frame_rate, extraction_method):
        empty_frames = AudioNodeBase.create_empty_tensor(audio, frame_rate, height, width, channels=3)
        feature_pipe = FeaturePipe(frame_rate, empty_frames)
        feature = AudioFeature(
            feature_name=extraction_method,
            audio=audio,
            frame_count=feature_pipe.frame_count,
            frame_rate=feature_pipe.frame_rate,
            feature_type=extraction_method
        )
        feature.extract()
        return (feature, feature_pipe, feature_pipe.frame_count)

class RhythmFeatureExtractor(FirstFeature):
    @classmethod
    def feature_type(cls) -> type[BaseFeature]:
        return RhythmFeature

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "audio": ("AUDIO",),
                "time_signature": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
            },
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "extract_feature"
    CATEGORY = "RyanOnTheInside/FlexFeatures/Audio/Rhythm"

    def extract_feature(self, audio, extraction_method, time_signature, frame_rate, video_frames):
        feature_pipe = FeaturePipe(frame_rate, video_frames)
        feature = RhythmFeature(
            feature_name=extraction_method,
            audio=audio,
            frame_count=feature_pipe.frame_count,
            frame_rate=feature_pipe.frame_rate,
            feature_type=extraction_method,
            time_signature=time_signature
        )
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
                "audio": ("AUDIO",),
                "feature_pipe": ("FEATURE_PIPE",),
                "opt_crepe_model":(["none", "medium", "tiny", "small", "large", "full"], {"default": "medium"})
            },
            "optional": {
                "opt_pitch_range_collections": ("PITCH_RANGE_COLLECTION",),
                # "": ("CREPE_MODEL",),
            },
        }

    RETURN_TYPES = ("FEATURE", "FEATURE_PIPE")
    FUNCTION = "extract_feature"

    def extract_feature(self, audio, feature_pipe, extraction_method, opt_pitch_range_collections=None, opt_crepe_model=None):
        if opt_pitch_range_collections is None:
            opt_pitch_range_collections = []
        feature = PitchFeature(
            feature_name=extraction_method,
            audio=audio,
            frame_count=feature_pipe.frame_count,
            frame_rate=feature_pipe.frame_rate,
            pitch_range_collections=opt_pitch_range_collections,
            feature_type=extraction_method,
            crepe_model=opt_crepe_model
        )
        feature.extract()
        return (feature, feature_pipe)

class PitchAbstraction(RyanOnTheInside):
    CATEGORY="RyanOnTheInside/FlexFeatures/Audio/Pitch"

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
                "pitch_tolerance_percent": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1}),
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