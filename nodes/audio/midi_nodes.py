import torch
import numpy as np
import mido
from .audio_nodes import AudioNodeBase
from ...tooltips import apply_tooltips

@apply_tooltips
class MIDIToAudio(AudioNodeBase):
    """Converts MIDI data to audio format compatible with ComfyUI's audio system."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi": ("MIDI",),
                "sample_rate": ("INT", {
                    "default": 44100,
                    "min": 8000,
                    "max": 192000,
                    "step": 1
                }),
                "instrument": ("INT", {
                    "default": 0,  # Piano (not used for simple synthesis)
                    "min": 0,
                    "max": 127,
                    "step": 1
                }),
                "volume": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                })
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "convert_midi_to_audio"
    CATEGORY = "RyanOnTheInside/Audio/MIDI"
    
    def convert_midi_to_audio(self, midi, sample_rate, instrument, volume):
        try:
            # Get tempo from MIDI file (default to 120 BPM if not specified)
            tempo = 500000  # Default tempo (microseconds per quarter note)
            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                        break
            
            # Calculate total duration in seconds
            total_time = 0
            for track in midi.tracks:
                track_time = 0
                for msg in track:
                    if hasattr(msg, 'time'):
                        track_time += msg.time
                total_time = max(total_time, track_time)
            
            # Convert ticks to seconds using tempo
            seconds_per_tick = tempo / (midi.ticks_per_beat * 1000000.0)
            total_time = total_time * seconds_per_tick
            
            # Ensure we have some duration
            total_time = max(total_time, 1.0)  # At least 1 second
            
            print(f"MIDI duration: {total_time:.2f} seconds")
            
            # Create empty audio buffer
            audio_length = int(total_time * sample_rate)
            audio_buffer = np.zeros(audio_length)
            
            # Process MIDI messages from all tracks
            for track in midi.tracks:
                current_time = 0
                active_notes = {}  # {note: (start_time, velocity)}
                
                for msg in track:
                    if hasattr(msg, 'time'):
                        current_time += msg.time * seconds_per_tick
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Start a new note
                        active_notes[msg.note] = (current_time, msg.velocity)
                    
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        # End a note
                        if msg.note in active_notes:
                            start_time, velocity = active_notes[msg.note]
                            duration = current_time - start_time
                            
                            # Generate note audio using simple sine wave
                            frequency = 440.0 * (2.0 ** ((msg.note - 69) / 12.0))  # Convert MIDI note to frequency
                            note_samples = int(duration * sample_rate)
                            if note_samples > 0:
                                t = np.linspace(0, duration, note_samples, False)
                                # Simple envelope to avoid clicks
                                envelope = np.ones_like(t)
                                attack = int(0.01 * sample_rate)
                                release = int(0.01 * sample_rate)
                                if len(envelope) > attack:
                                    envelope[:attack] = np.linspace(0, 1, attack)
                                if len(envelope) > release:
                                    envelope[-release:] = np.linspace(1, 0, release)
                                note_audio = np.sin(2 * np.pi * frequency * t) * envelope * (velocity / 127.0) * volume
                                
                                # Add to buffer with proper timing
                                start_idx = int(start_time * sample_rate)
                                end_idx = start_idx + len(note_audio)
                                if end_idx > len(audio_buffer):
                                    audio_buffer = np.pad(audio_buffer, (0, end_idx - len(audio_buffer)))
                                audio_buffer[start_idx:end_idx] += note_audio
                            
                            del active_notes[msg.note]
            
            # Normalize audio if we have any content
            if len(audio_buffer) > 0 and np.max(np.abs(audio_buffer)) > 0:
                audio_buffer = audio_buffer / np.max(np.abs(audio_buffer))
            
            # Convert to stereo
            audio_buffer = np.stack([audio_buffer, audio_buffer])
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_buffer).float()
            
            print(f"Generated audio length: {len(audio_buffer[0])/sample_rate:.2f} seconds")
            
            return ({
                "waveform": audio_tensor.unsqueeze(0),  # Add batch dimension
                "sample_rate": sample_rate
            },)
            
        except Exception as e:
            # If anything goes wrong, return empty audio
            print(f"Error in MIDIToAudio: {str(e)}")
            empty_audio = torch.zeros((1, 2, 1))  # 1 batch, 2 channels, 1 sample
            return ({
                "waveform": empty_audio,
                "sample_rate": sample_rate
            },)

NODE_CLASS_MAPPINGS = {
    "MIDIToAudio": MIDIToAudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDIToAudio": "MIDI to Audio"
} 