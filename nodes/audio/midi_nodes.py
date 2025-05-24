import torch
import numpy as np
import mido
import os
import folder_paths
import math
from .audio_nodes import AudioNodeBase
from ...tooltips import apply_tooltips
from server import PromptServer
from aiohttp import web
import shutil

@apply_tooltips
class MIDIToAudio(AudioNodeBase):
    """Converts MIDI data to audio format compatible with ComfyUI's audio system."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi": ("MIDI",),
                "instrument_type": (["Piano", "Synth", "Bass", "Drums"],),
                "sample_rate": ("INT", {
                    "default": 44100,
                    "min": 8000,
                    "max": 192000,
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
    DESCRIPTION = "This will produce rudimentary approximations of MIDI notes for quick reference and nothing more. May or may not get the notes right, especially for drums."
    
    def generate_instrument_waveform(self, frequency, t, instrument_type, envelope, note_number=60):
        """Generate waveform based on the selected instrument type"""
        if instrument_type == "Piano":
            # Piano-like sound with some harmonics
            waveform = 0.5 * np.sin(2 * np.pi * frequency * t)  # Fundamental
            waveform += 0.2 * np.sin(2 * np.pi * frequency * 2 * t)  # 1st harmonic (octave)
            waveform += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)  # 2nd harmonic
            waveform += 0.05 * np.sin(2 * np.pi * frequency * 4 * t)  # 3rd harmonic
            # Add faster decay for piano-like sound
            decay = np.exp(-t * 3)
            return waveform * envelope * decay
        
        elif instrument_type == "Bass":
            # Bass with more low frequencies
            waveform = 0.6 * np.sin(2 * np.pi * frequency * t)  # Fundamental
            waveform += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)  # 1st harmonic
            # Add some distortion for bass character
            waveform = np.tanh(waveform * 1.5) * 0.7
            # Slower decay for bass
            decay = np.exp(-t * 2)
            return waveform * envelope * decay
        
        elif instrument_type == "Drums":
            # Use standard MIDI drum mappings (channel 10)
            # Define custom synthesis for each drum sound based on note number
            
            # Bass/Kick Drums
            if note_number in [35, 36]:  # Bass Drum 2, Bass Drum 1
                # Low frequency sine with very fast decay
                waveform = np.sin(2 * np.pi * 60 * t)  # Fixed low frequency for kick
                waveform += 0.2 * np.sin(2 * np.pi * 90 * t)  # Add some mid tone
                decay = np.exp(-t * 20)  # Very quick decay
                return waveform * envelope * decay
            
            # Snare Drums
            elif note_number in [38, 40]:  # Acoustic Snare, Electric Snare
                # Mix of sine wave and noise
                waveform = 0.3 * np.sin(2 * np.pi * 150 * t)  # Mid frequency tone
                noise = np.random.rand(len(t)) * 2 - 1  # White noise
                decay = np.exp(-t * 15)  # Fast decay
                return (waveform + 0.7 * noise) * envelope * decay
            
            # Hi-Hats
            elif note_number in [42, 44, 46]:  # Closed, Pedal, Open Hi-Hats
                # Mostly noise with different decay times
                noise = np.random.rand(len(t)) * 2 - 1
                # Different decay times based on hi-hat type
                if note_number == 42:  # Closed Hi-Hat
                    decay = np.exp(-t * 30)  # Very short
                elif note_number == 44:  # Pedal Hi-Hat
                    decay = np.exp(-t * 25)  # Short
                else:  # Open Hi-Hat
                    decay = np.exp(-t * 10)  # Longer
                
                # Add some high frequency sine for metallic character
                waveform = noise + 0.1 * np.sin(2 * np.pi * 800 * t)
                
                return waveform * envelope * decay
            
            # Toms
            elif note_number in [41, 43, 45, 47, 48, 50]:  # Various Toms
                # Pitched sine waves with medium decay
                if note_number in [41, 43]:  # Floor Toms
                    tom_freq = 80  # Low frequency
                elif note_number in [45, 47]:  # Mid Toms
                    tom_freq = 120  # Mid frequency
                else:  # High Toms
                    tom_freq = 180  # Higher frequency
                
                waveform = np.sin(2 * np.pi * tom_freq * t)
                decay = np.exp(-t * 12)
                return waveform * envelope * decay
            
            # Cymbals
            elif note_number in [49, 51, 52, 53, 55, 57]:  # Crash and Ride Cymbals
                # Complex noise with slow decay
                noise = np.random.rand(len(t)) * 2 - 1
                # Add some high frequencies for metallic sound
                for i in range(3, 10):
                    noise += 0.1 / i * np.sin(2 * np.pi * 500 * i * t)
                
                # Longer decay for cymbals
                decay = np.exp(-t * 4)
                return noise * envelope * decay
            
            # Other percussion (default case)
            else:
                # Generic percussion sound based on frequency
                noise = np.random.rand(len(t)) * 2 - 1
                waveform = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.5 * noise
                decay = np.exp(-t * 10)
                return waveform * envelope * decay
        
        else:  # "Synth" (default)
            # Simple synth with multiple waveforms
            waveform = 0.4 * np.sin(2 * np.pi * frequency * t)  # Sine
            # Add square wave component
            square = 0.3 * np.sign(np.sin(2 * np.pi * frequency * t))
            # Add sawtooth component
            sawtooth = 0.3 * ((2 * (frequency * t - np.floor(0.5 + frequency * t))) % 2)
            
            return (waveform + square + sawtooth) * envelope
    
    def convert_midi_to_audio(self, midi, instrument_type, sample_rate, volume):
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
            last_event_time = 0
            has_notes = False
            
            for track in midi.tracks:
                track_time = 0
                notes_in_track = False
                
                for msg in track:
                    if hasattr(msg, 'time'):
                        track_time += msg.time
                        if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                            notes_in_track = True
                            has_notes = True
                            last_event_time = track_time
                        elif (msg.type == 'note_off' or (msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity == 0)):
                            last_event_time = track_time
                
                total_time = max(total_time, track_time)
            
            # If we found notes, use the last note event time plus some padding
            if has_notes:
                total_time = last_event_time + 240  # Add a small buffer (240 ticks) for note releases
            
            # Convert ticks to seconds using tempo
            seconds_per_tick = tempo / (midi.ticks_per_beat * 1000000.0)
            total_time = total_time * seconds_per_tick
            
            # Ensure we have some duration
            total_time = max(total_time, 0.5)  # At least 0.5 second
            
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
                            
                            # Generate note audio using instrument-specific synthesis
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
                                
                                # Generate waveform based on instrument type
                                note_audio = self.generate_instrument_waveform(
                                    frequency, t, instrument_type, envelope, msg.note
                                ) * (velocity / 127.0) * volume
                                
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

def calculate_midi_total_measures(midi_data):
    """Calculate the total number of measures in the MIDI file"""
    # Get time signature (defaults to 4/4)
    time_sig_numerator = 4
    time_sig_denominator = 4
    
    for track in midi_data.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                time_sig_numerator = msg.numerator
                time_sig_denominator = msg.denominator
                break
    
    # Calculate total ticks
    total_ticks = 0
    for track in midi_data.tracks:
        track_ticks = 0
        for msg in track:
            if hasattr(msg, 'time'):
                track_ticks += msg.time
        total_ticks = max(total_ticks, track_ticks)
    
    # Calculate ticks per measure
    ticks_per_beat = midi_data.ticks_per_beat
    ticks_per_measure = ticks_per_beat * 4 * (time_sig_numerator / time_sig_denominator)
    
    # Calculate total measures
    total_measures = math.ceil(total_ticks / ticks_per_measure)
    return total_measures, time_sig_numerator, time_sig_denominator

def convert_measures_to_ticks_range(midi_data, start_measure, start_beat, end_measure, end_beat):
    """Convert musical measure range to MIDI tick range"""
    # Get time signature (defaults to 4/4)
    time_sig_numerator = 4
    time_sig_denominator = 4
    
    for track in midi_data.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                time_sig_numerator = msg.numerator
                time_sig_denominator = msg.denominator
                break
    
    # Calculate ticks per beat and measure
    ticks_per_beat = midi_data.ticks_per_beat
    beats_per_measure = time_sig_numerator
    
    # Adjust start measure/beat to 0-indexed for calculation
    start_measure_index = start_measure - 1
    start_beat_index = start_beat - 1
    
    # Calculate start tick
    start_tick = (start_measure_index * beats_per_measure + start_beat_index) * ticks_per_beat
    
    # Calculate end tick
    if end_measure > 0:
        end_measure_index = end_measure - 1
        end_beat_index = end_beat - 1
        end_tick = (end_measure_index * beats_per_measure + end_beat_index) * ticks_per_beat
    else:
        end_tick = float('inf')
    
    return start_tick, end_tick

@apply_tooltips
class MIDILoader:
    """Loads MIDI files for processing in ComfyUI with options for selecting specific measures."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_file": (folder_paths.get_filename_list("midi_files"),),
                "track_selection": (["all"],),
                "start_measure": ("INT", {"default": 1, "min": 1, "step": 1}),
                "start_beat": ("INT", {"default": 1, "min": 1, "step": 1}),
                "end_measure": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "End measure (0 = all remaining measures)"}),
                "end_beat": ("INT", {"default": 1, "min": 1, "step": 1, "tootip": "End beat"})
            }
        }

    RETURN_TYPES = ("MIDI",)
    FUNCTION = "load_midi"
    CATEGORY = "RyanOnTheInside/Audio/MIDI"
    
    def load_midi(self, midi_file, track_selection, start_measure=1, start_beat=1, end_measure=0, end_beat=1):
        try:
            midi_path = folder_paths.get_full_path("midi_files", midi_file)
            if not midi_path or not os.path.exists(midi_path):
                raise FileNotFoundError(f"MIDI file not found: {midi_file}")

            # Load the MIDI file
            midi_data = mido.MidiFile(midi_path)
            
            # Apply track selection if not "all"
            if track_selection != "all":
                track_index_str = track_selection.split(':')[0].strip()
                if track_index_str.isdigit():
                    track_index = int(track_index_str)
                    # Create a new MIDI file with only the selected track
                    selected_midi = mido.MidiFile(ticks_per_beat=midi_data.ticks_per_beat)
                    
                    # First, add any metadata tracks (typically track 0 in type 1 MIDI files)
                    if midi_data.type == 1 and track_index > 0:
                        selected_midi.tracks.append(midi_data.tracks[0])
                    
                    # Add the selected track
                    if track_index < len(midi_data.tracks):
                        selected_midi.tracks.append(midi_data.tracks[track_index])
                    
                    midi_data = selected_midi
            
            # Apply measure slicing if needed
            if (start_measure > 1 or start_beat > 1 or end_measure > 0) and midi_data.tracks:
                # Get total measures in the file for validation
                total_measures, _, _ = calculate_midi_total_measures(midi_data)
                
                # If end measure is 0, use the total measures
                if end_measure == 0:
                    end_measure = total_measures
                
                # Convert musical measures to ticks
                start_tick, end_tick = convert_measures_to_ticks_range(midi_data, start_measure, start_beat, end_measure, end_beat)
                
                # The total duration in ticks is EXACTLY end_tick - start_tick
                total_tick_duration = end_tick - start_tick
                
                # Create a new MIDI file with the selected measures
                trimmed_midi = mido.MidiFile(ticks_per_beat=midi_data.ticks_per_beat)
                
                for track in midi_data.tracks:
                    new_track = mido.MidiTrack()
                    trimmed_midi.tracks.append(new_track)
                    
                    # Copy important metadata messages first
                    for msg in track:
                        if not hasattr(msg, 'time'):
                            if msg.type in ['track_name', 'time_signature', 'key_signature', 'set_tempo']:
                                new_track.append(msg.copy())
                    
                    # Collect events within the time range
                    note_events = []
                    active_notes = {}  # {note_num: tick_time_started}
                    notes_active_before_range = {}  # Track notes that started before our range
                    current_tick = 0
                    
                    # First pass: collect all note on/off events, tracking notes that cross boundaries
                    for msg in track:
                        if not hasattr(msg, 'time'):
                            continue
                        
                        current_tick += msg.time
                        
                        # Handle note_on events
                        if msg.type == 'note_on' and msg.velocity > 0:
                            # Note starting before our range
                            if current_tick < start_tick:
                                notes_active_before_range[msg.note] = current_tick
                            # Note starting within our range
                            elif current_tick <= end_tick:
                                active_notes[msg.note] = current_tick
                                note_events.append((current_tick, msg.copy()))
                        
                        # Handle note_off events
                        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                            # Handle notes that started before our range but end within it
                            if msg.note in notes_active_before_range:
                                if current_tick >= start_tick and current_tick <= end_tick:
                                    # Create a new note_on at the start boundary
                                    new_note_on = mido.Message('note_on', note=msg.note, velocity=64, time=0)
                                    note_events.append((start_tick, new_note_on))
                                    # Add the note_off event
                                    note_events.append((current_tick, msg.copy()))
                                del notes_active_before_range[msg.note]
                            
                            # Handle notes that started within our range
                            elif msg.note in active_notes:
                                # If note ends within our range, add the note_off event
                                if current_tick <= end_tick:
                                    note_events.append((current_tick, msg.copy()))
                                # If note would end outside our range, add a note_off at the end boundary
                                else:
                                    new_note_off = mido.Message('note_off', note=msg.note, velocity=0, time=0)
                                    note_events.append((end_tick, new_note_off))
                                del active_notes[msg.note]
                        
                        # Include other MIDI events within the time range
                        elif current_tick >= start_tick and current_tick <= end_tick:
                            if msg.type in ['set_tempo', 'control_change', 'program_change', 'pitchwheel']:
                                note_events.append((current_tick, msg.copy()))
                    
                    # Add note_off events at end boundary for any notes still active at the end
                    for note in active_notes:
                        new_note_off = mido.Message('note_off', note=note, velocity=0, time=0)
                        note_events.append((end_tick, new_note_off))
                    
                    # Sort events by tick time
                    note_events.sort(key=lambda x: x[0])
                    
                    # Add events to the new track with adjusted timing
                    prev_tick = start_tick
                    
                    for tick, msg in note_events:
                        # Adjust timing relative to previous event
                        adjusted_msg = msg.copy()
                        adjusted_msg.time = tick - prev_tick
                        prev_tick = tick
                        
                        new_track.append(adjusted_msg)
                    
                    # Calculate how many ticks remain until the exact end_tick
                    remaining_ticks = start_tick + total_tick_duration - prev_tick
                    
                    # Add end of track marker at exactly the right position
                    end_track = mido.MetaMessage('end_of_track')
                    end_track.time = max(0, remaining_ticks)
                    new_track.append(end_track)
                
                midi_data = trimmed_midi
            
            return (midi_data,)

        except Exception as e:
            raise RuntimeError(f"Error loading MIDI file: {type(e).__name__}: {str(e)}")
    
    @classmethod
    def analyze_midi(cls, midi_path, start_measure=1, start_beat=1, end_measure=0, end_beat=1):
        midi_data = mido.MidiFile(midi_path)
        
        # Get the total measures
        total_measures, time_sig_num, time_sig_denom = calculate_midi_total_measures(midi_data)
        
        # If end measure is 0, use the total measures
        if end_measure == 0:
            end_measure = total_measures
        
        # Convert measures to ticks if measure filtering is applied
        if start_measure > 1 or start_beat > 1 or end_measure < total_measures:
            start_tick, end_tick = convert_measures_to_ticks_range(midi_data, start_measure, start_beat, end_measure, end_beat)
        else:
            start_tick = 0
            end_tick = float('inf')
        
        tracks = ["all"]
        all_notes = set()
        track_notes = {}
        for i, track in enumerate(midi_data.tracks):
            track_notes[str(i)] = set()
            current_tick = 0
            
            # Keep track of notes that start within range
            active_notes = set()
            
            for msg in track:
                if hasattr(msg, 'time'):
                    current_tick += msg.time
                
                # Only consider notes within our measure range
                if current_tick >= start_tick and current_tick <= end_tick:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        track_notes[str(i)].add(msg.note)
                        all_notes.add(msg.note)
                        active_notes.add(msg.note)
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in active_notes:
                            active_notes.remove(msg.note)
            
            if len(track_notes[str(i)]) == 0:
                tracks.append(f"{i}: (Empty)")
            else:
                tracks.append(f"{i}: {getattr(track, 'name', '') or f'Track {i}'}")
        
        return {
            "tracks": tracks,
            "all_notes": ",".join(map(str, sorted(set(all_notes)))),
            "track_notes": {k: ",".join(map(str, sorted(v))) for k, v in track_notes.items()},
            "total_measures": total_measures,
            "time_signature": f"{time_sig_num}/{time_sig_denom}"
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, midi_file, track_selection, start_measure=1, start_beat=1, end_measure=0, end_beat=1):
        midi_path = folder_paths.get_full_path("midi_files", midi_file)
        if not midi_path or not os.path.isfile(midi_path):
            return f"MIDI file not found: {midi_file}"
        
        # Check if the file has a .mid or .midi extension
        if not midi_file.lower().endswith(('.mid', '.midi')):
            return f"Invalid file type. Expected .mid or .midi file, got: {midi_file}"
        
        if start_measure < 1:
            return f"Start measure must be at least 1, got: {start_measure}"
        
        if start_beat < 1:
            return f"Start beat must be at least 1, got: {start_beat}"
        
        if end_measure < 0:
            return f"End measure cannot be negative, got: {end_measure}"
        
        if end_beat < 1:
            return f"End beat must be at least 1, got: {end_beat}"
        
        # Check if the start measure is valid for this MIDI file
        try:
            midi_data = mido.MidiFile(midi_path)
            total_measures, _, _ = calculate_midi_total_measures(midi_data)
            
            if start_measure > total_measures:
                return f"Start measure {start_measure} exceeds the total measures in the file ({total_measures})"
            
            if end_measure > 0 and end_measure > total_measures:
                return f"End measure {end_measure} exceeds the total measures in the file ({total_measures})"
            
            if end_measure > 0 and end_measure < start_measure:
                return f"End measure {end_measure} cannot be less than start measure {start_measure}"
                
            if end_measure == start_measure and end_beat < start_beat:
                return f"When start and end measures are the same, end beat {end_beat} cannot be less than start beat {start_beat}"
        except Exception as e:
            return f"Error validating MIDI file: {str(e)}"
        
        return True

# Server routes for MIDI file handling
routes = PromptServer.instance.routes

@routes.post('/get_track_notes')
async def get_track_notes(request):
    data = await request.json()
    midi_file = data.get('midi_file')
    start_measure = data.get('start_measure', 1)
    start_beat = data.get('start_beat', 1)
    end_measure = data.get('end_measure', 0)
    end_beat = data.get('end_beat', 1)

    if not midi_file:
        return web.json_response({"error": "Missing required parameters"}, status=400)

    midi_path = folder_paths.get_full_path("midi_files", midi_file)
    if not midi_path or not os.path.exists(midi_path):
        return web.json_response({"error": "MIDI file not found"}, status=404)

    # Analyze MIDI file
    analysis = MIDILoader.analyze_midi(midi_path, start_measure, start_beat, end_measure, end_beat)
    return web.json_response(analysis)

@routes.post('/upload_midi')
async def upload_midi(request):
    data = await request.post()
    midi_file = data['file']
    
    if midi_file and midi_file.filename:
        safe_filename = os.path.basename(midi_file.filename)
        
        midi_dir = folder_paths.get_folder_paths("midi_files")[0]
        
        if not os.path.exists(midi_dir):
            os.makedirs(midi_dir, exist_ok=True)
        
        midi_path = os.path.join(midi_dir, safe_filename)

        with open(midi_path, 'wb') as f:
            shutil.copyfileobj(midi_file.file, f)

        midi_files = folder_paths.get_filename_list("midi_files")
        analysis = MIDILoader.analyze_midi(midi_path)

        return web.json_response({
            "status": "success",
            "uploaded_file": safe_filename,
            "midi_files": midi_files,
            "analysis": analysis
        })
    else:
        return web.json_response({"status": "error", "message": "No file uploaded"}, status=400)
    
@routes.post('/refresh_midi_data')
async def refresh_midi_data(request):
    data = await request.json()
    midi_file = data.get('midi_file')
    track_selection = data.get('track_selection')
    start_measure = data.get('start_measure', 1)
    start_beat = data.get('start_beat', 1)
    end_measure = data.get('end_measure', 0)
    end_beat = data.get('end_beat', 1)

    if not midi_file:
        return web.json_response({"error": "Missing required parameters"}, status=400)

    midi_path = folder_paths.get_full_path("midi_files", midi_file)
    if not midi_path or not os.path.exists(midi_path):
        return web.json_response({"error": "MIDI file not found"}, status=404)

    # Load the full MIDI data
    midi_data = mido.MidiFile(midi_path)
    
    # Get total measures information
    total_measures, time_sig_num, time_sig_denom = calculate_midi_total_measures(midi_data)
    
    # If end measure is 0, use the total measures
    if end_measure == 0:
        end_measure = total_measures
    
    # Apply track selection and measure filtering
    all_notes = set()
    
    # Convert measures to ticks
    if start_measure > 1 or start_beat > 1 or end_measure < total_measures:
        start_tick, end_tick = convert_measures_to_ticks_range(midi_data, start_measure, start_beat, end_measure, end_beat)
    else:
        start_tick = 0
        end_tick = float('inf')
    
    # Process tracks to collect notes within the time range
    track_notes = {}
    for i, track in enumerate(midi_data.tracks):
        track_notes[str(i)] = set()
        current_tick = 0
        
        # Keep track of notes that start within range
        active_notes = set()
        
        for msg in track:
            if hasattr(msg, 'time'):
                current_tick += msg.time
                
            # Only consider note_on messages with velocity > 0 (actual notes being played)
            if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                # Check if this note is within our measure range
                if current_tick >= start_tick and current_tick < end_tick:
                    note = msg.note
                    track_notes[str(i)].add(note)
                    active_notes.add(note)
                    # If all tracks are selected or the current track matches selection, add to all_notes
                    if track_selection == "all" or track_selection.startswith(f"{i}:"):
                        all_notes.add(note)
            
            # Remove notes from active set when they're turned off, but don't remove from track_notes
            elif msg.type == 'note_off' or (msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity == 0):
                if msg.note in active_notes:
                    active_notes.remove(msg.note)
    
    # If a specific track is selected but we didn't find any notes, it might be because
    # track selection was changed after note filtering
    if track_selection != "all" and not all_notes:
        track_index = track_selection.split(':')[0]
        if track_index.isdigit() and track_index in track_notes:
            all_notes = track_notes[track_index]
    
    # Format and return the response
    return web.json_response({
        "tracks": ["all"] + [f"{i}: {getattr(track, 'name', '') or f'Track {i}'}" for i, track in enumerate(midi_data.tracks)],
        "all_notes": ",".join(map(str, sorted(all_notes))),
        "track_notes": {k: ",".join(map(str, sorted(v))) for k, v in track_notes.items()},
        "total_measures": total_measures,
        "time_signature": f"{time_sig_num}/{time_sig_denom}"
    })

NODE_CLASS_MAPPINGS = {
    "MIDIToAudio": MIDIToAudio,
    "MIDILoader": MIDILoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDIToAudio": "MIDI to Audio",
    "MIDILoader": "MIDI Loader"
}