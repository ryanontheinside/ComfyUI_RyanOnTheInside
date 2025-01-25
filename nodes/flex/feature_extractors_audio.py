from .feature_extractors import FeatureExtractorBase
from .features_audio import AudioFeature, PitchFeature, PitchRange, BaseFeature, RhythmFeature
from ... import RyanOnTheInside
from ..audio.audio_nodes import AudioNodeBase
from ...tooltips import apply_tooltips

_category = f"{FeatureExtractorBase.CATEGORY}/Audio"

class AudioFeatureExtractorMixin:
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["frame_count"] = ("INT", {"default": 0, "min": 0})
        return {
            **super().INPUT_TYPES(),
            "required": {
                **parent_inputs,
                "audio": ("AUDIO",),
            }
        }

    def calculate_target_frame_count(self, audio, frame_rate, frame_count):
        """Calculate the target frame count based on audio length and specified frame count"""
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        natural_frame_count = int((waveform.shape[-1] / sample_rate) * frame_rate)
        return frame_count if frame_count > 0 else natural_frame_count

@apply_tooltips
class AudioFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (AudioFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("FEATURE", "INT",)
    RETURN_NAMES = ("feature", "frame_count",)
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(self, audio, frame_rate, frame_count, width, height, extraction_method):
        target_frame_count = self.calculate_target_frame_count(audio, frame_rate, frame_count)

        feature = AudioFeature(
            width=width,
            height=height,
            feature_name=extraction_method,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            feature_type=extraction_method
        )
        feature.extract()
        return (feature, target_frame_count)

@apply_tooltips
class RhythmFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (RhythmFeature.get_extraction_methods(),)
        return {
            "required": {
                **parent_inputs,
                "audio": ("AUDIO",),
                "time_signature": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
            },
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(self, audio, extraction_method, time_signature, frame_rate, frame_count, width, height):
        target_frame_count = self.calculate_target_frame_count(audio, frame_rate, frame_count)

        feature = RhythmFeature(
            width=width,
            height=height,
            feature_name=extraction_method,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            feature_type=extraction_method,
            time_signature=time_signature
        )
        feature.extract()
        return (feature,)

@apply_tooltips
class PitchFeatureExtractor(AudioFeatureExtractorMixin, FeatureExtractorBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (PitchFeature.get_extraction_methods(),)
        return {            
            "required": {
                **parent_inputs,
                "audio": ("AUDIO",),
                "opt_crepe_model":(["none", "medium", "tiny", "small", "large", "full"], {"default": "medium"})
            },
            "optional": {
                "opt_pitch_range_collections": ("PITCH_RANGE_COLLECTION",),
            },
        }

    RETURN_TYPES = ("FEATURE",)
    FUNCTION = "extract_feature"
    CATEGORY = _category

    def extract_feature(self, audio, frame_rate, frame_count, width, height, extraction_method, opt_pitch_range_collections=None, opt_crepe_model=None):
        if opt_pitch_range_collections is None:
            opt_pitch_range_collections = []

        target_frame_count = self.calculate_target_frame_count(audio, frame_rate, frame_count)

        feature = PitchFeature(
            width=width,
            height=height,
            feature_name=extraction_method,
            audio=audio,
            frame_count=target_frame_count,
            frame_rate=frame_rate,
            pitch_range_collections=opt_pitch_range_collections,
            feature_type=extraction_method,
            crepe_model=opt_crepe_model
        )
        feature.extract()
        return (feature,)

class PitchAbstraction(RyanOnTheInside):
    CATEGORY="RyanOnTheInside/FlexFeatures/Audio/Pitch"

@apply_tooltips
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
    CATEGORY = _category

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
    
@apply_tooltips
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
    CATEGORY = _category

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
    
@apply_tooltips
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
    CATEGORY = _category

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
