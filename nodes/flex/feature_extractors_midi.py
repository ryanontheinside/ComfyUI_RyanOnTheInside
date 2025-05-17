import mido
import os
import folder_paths
from .features_midi import MIDIFeature
from .feature_extractors import FeatureExtractorBase
from ...tooltips import apply_tooltips

@apply_tooltips
class MIDIFeatureExtractor(FeatureExtractorBase):
    """Extract musical features from MIDI data based on selected notes from a piano interface.
    
    This node connects to a MIDILoader node to receive MIDI data and extract features 
    based on selected notes. Features can be used to modulate other parameters in the workflow.
    """
    
    @classmethod
    def feature_type(cls) -> type[MIDIFeature]:
        return MIDIFeature

    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()["required"]
        parent_inputs["extraction_method"] = (MIDIFeature.get_extraction_methods(),)
        parent_inputs["frame_count"] = ("INT", {"default": 0, "min": 0})  # Changed default to 0 for auto-calculation
        return {
            "required": {
                **parent_inputs,
                "midi": ("MIDI",),
                "chord_only": ("BOOLEAN", {"default": False}),
                "notes":  ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MIDI", "FEATURE", "INT")
    RETURN_NAMES = ("midi", "feature", "frame_count")
    FUNCTION = "process_midi"
    CATEGORY = "RyanOnTheInside/Audio/Features"

    def calculate_target_frame_count(self, midi, frame_rate, frame_count):
        """Calculate the target frame count based on MIDI duration and specified frame count"""
        if frame_count > 0:
            return frame_count
        
        # Calculate total duration in seconds
        total_time = 0
        for track in midi.tracks:
            track_time = 0
            for msg in track:
                if hasattr(msg, 'time'):
                    track_time += msg.time
            total_time = max(total_time, track_time)
        
        # Get tempo (default to 120 BPM if not specified)
        tempo = 500000  # Default tempo in microseconds per quarter note
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        
        # Convert ticks to seconds
        seconds_per_tick = tempo / (midi.ticks_per_beat * 1000000.0)
        duration_seconds = total_time * seconds_per_tick
        
        # Ensure we have some duration
        duration_seconds = max(duration_seconds, 1.0)  # At least 1 second
        
        # Calculate frame count
        return max(1, int(duration_seconds * frame_rate))

    def process_midi(self, midi, notes, extraction_method, frame_rate, frame_count, width, height, chord_only=False):
        try:
            # Calculate appropriate frame count
            target_frame_count = self.calculate_target_frame_count(midi, frame_rate, frame_count)
            
            selected_notes = [int(n.strip()) for n in notes.split(',') if n.strip().isdigit()]
            
            # Convert friendly attribute name to internal attribute name
            internal_attribute = MIDIFeature.get_attribute_value(extraction_method)
            
            feature = MIDIFeature(
                f"midi_{internal_attribute}",
                midi,
                internal_attribute,
                frame_rate,
                target_frame_count,  # Use calculated frame count
                width,
                height,
                notes=selected_notes,
                chord_only=chord_only
            )
            
            feature.extract()

            return (midi, feature, target_frame_count)

        except Exception as e:
            raise RuntimeError(f"Error processing MIDI: {type(e).__name__}: {str(e)}")
    
    @classmethod
    def VALIDATE_INPUTS(cls, midi, notes, extraction_method, frame_rate, frame_count, width, height, chord_only=False):
        if notes:
            try:
                note_list = [int(n.strip()) for n in notes.split(',') if n.strip()]
                if not all(0 <= n <= 127 for n in note_list):
                    return "Invalid note value. All notes must be between 0 and 127."
            except ValueError:
                return "Invalid notes format. Please provide comma-separated integers."
        
        return True
            

NODE_CLASS_MAPPINGS = {
    "MIDIFeatureExtractor": MIDIFeatureExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDIFeatureExtractor": "MIDI Feature Extractor",
}