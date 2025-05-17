import torch
import numpy as np
import mido
import os
import folder_paths
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

@apply_tooltips
class MIDILoader:
    """Loads MIDI files for processing in ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_file": (folder_paths.get_filename_list("midi_files"),),
                "track_selection": (["all"],),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1, "description": "Duration in seconds (0 = full duration)"}),
            }
        }

    RETURN_TYPES = ("MIDI",)
    FUNCTION = "load_midi"
    CATEGORY = "RyanOnTheInside/Audio/MIDI"
    
    def load_midi(self, midi_file, track_selection, start_time=0.0, duration=0.0):
        try:
            midi_path = folder_paths.get_full_path("midi_files", midi_file)
            if not midi_path or not os.path.exists(midi_path):
                raise FileNotFoundError(f"MIDI file not found: {midi_file}")

            # Load the MIDI file
            midi_data = mido.MidiFile(midi_path)
            
            # Get tempo for time calculations
            tempo = 500000  # Default tempo (microseconds per quarter note)
            for track in midi_data.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                        break
            
            # Calculate seconds per tick for time conversion
            seconds_per_tick = tempo / (midi_data.ticks_per_beat * 1000000.0)
            
            # Convert seconds to ticks
            start_tick = int(start_time / seconds_per_tick) if start_time > 0 else 0
            end_tick = float('inf')
            if duration > 0:
                end_tick = start_tick + int(duration / seconds_per_tick)
            
            # Calculate total duration
            total_ticks = 0
            for track in midi_data.tracks:
                track_ticks = 0
                for msg in track:
                    if hasattr(msg, 'time'):
                        track_ticks += msg.time
                total_ticks = max(total_ticks, track_ticks)
            
            total_duration = total_ticks * seconds_per_tick
            
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
            
            # Apply time slicing if needed
            if (start_time > 0 or duration > 0) and midi_data.tracks:
                # Create a new MIDI file with the selected time range
                trimmed_midi = mido.MidiFile(ticks_per_beat=midi_data.ticks_per_beat)
                
                # Copy data for the trimmed section
                # NOTE: This is a simplified approach and may not handle all MIDI events perfectly
                for track in midi_data.tracks:
                    new_track = mido.MidiTrack()
                    trimmed_midi.tracks.append(new_track)
                    
                    # Copy all metadata and control messages
                    for msg in track:
                        if not hasattr(msg, 'time') or msg.type in ['time_signature', 'key_signature', 'set_tempo']:
                            new_track.append(msg.copy())
                    
                    # Extract all note events within our time range
                    current_tick = 0
                    last_tick = 0
                    
                    # First pass: collect all note_on events within our range
                    active_notes = set()
                    for msg in track:
                        if not hasattr(msg, 'time'):
                            continue
                            
                        current_tick += msg.time
                        
                        # Track note_on events within our range
                        if msg.type == 'note_on' and msg.velocity > 0:
                            if current_tick >= start_tick and current_tick < end_tick:
                                active_notes.add(msg.note)
                    
                    # Second pass: include all relevant events
                    current_tick = 0
                    for msg in track:
                        if not hasattr(msg, 'time'):
                            continue
                            
                        prev_tick = current_tick
                        current_tick += msg.time
                        
                        # Include message if:
                        # 1. It's within our time range
                        # 2. OR it's a note_off for an active note
                        # 3. OR it's a control message
                        in_range = current_tick >= start_tick and current_tick < end_tick
                        is_note_release = msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)
                        is_active_note = hasattr(msg, 'note') and msg.note in active_notes
                        is_control = msg.type in ['control_change', 'program_change', 'pitchwheel']
                        
                        if in_range or (is_note_release and is_active_note) or is_control:
                            new_msg = msg.copy()
                            
                            # Adjust timing for the first event
                            if not new_track or current_tick == start_tick:
                                new_msg.time = 0
                            else:
                                new_msg.time = current_tick - prev_tick
                                
                            new_track.append(new_msg)
                            
                            # Remove from active notes if it's a note off
                            if is_note_release and is_active_note:
                                active_notes.remove(msg.note)
                
                # If we actually need to trim the audio
                if start_time > 0 or duration > 0:
                    midi_data = trimmed_midi
            
            return (midi_data,)

        except Exception as e:
            raise RuntimeError(f"Error loading MIDI file: {type(e).__name__}: {str(e)}")
    
    @classmethod
    def analyze_midi(cls, midi_path, start_time=0.0, duration=0.0):
        """Analyze the MIDI file and return available notes within the time range"""
        midi_data = mido.MidiFile(midi_path)
        
        # Get tempo for time calculations
        tempo = 500000  # Default tempo (microseconds per quarter note)
        for track in midi_data.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
        
        # Calculate seconds per tick for time conversion
        seconds_per_tick = tempo / (midi_data.ticks_per_beat * 1000000.0)
        
        # Convert seconds to ticks
        start_tick = int(start_time / seconds_per_tick) if start_time > 0 else 0
        end_tick = float('inf')
        if duration > 0:
            end_tick = start_tick + int(duration / seconds_per_tick)
        
        # Collect track data and note data within time range
        tracks = ["all"]
        all_notes = set()
        track_notes = {}
        
        for i, track in enumerate(midi_data.tracks):
            track_notes[str(i)] = set()
            current_tick = 0
            
            for msg in track:
                if hasattr(msg, 'time'):
                    current_tick += msg.time
                
                # Only collect notes in our time range
                if current_tick >= start_tick and current_tick < end_tick:
                    if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                        track_notes[str(i)].add(msg.note)
                        all_notes.add(msg.note)
            
            # Add track name info
            track_name = getattr(track, 'name', None) or f'Track {i}'
            if len(track_notes[str(i)]) == 0:
                tracks.append(f"{i}: (Empty)")
            else:
                tracks.append(f"{i}: {track_name}")
        
        # Calculate total duration
        total_ticks = 0
        for track in midi_data.tracks:
            track_ticks = 0
            for msg in track:
                if hasattr(msg, 'time'):
                    track_ticks += msg.time
            total_ticks = max(total_ticks, track_ticks)
        
        total_duration = total_ticks * seconds_per_tick
        
        return {
            "tracks": tracks,
            "all_notes": ",".join(map(str, sorted(all_notes))),
            "track_notes": {k: ",".join(map(str, sorted(v))) for k, v in track_notes.items()},
            "total_duration": total_duration,
            "selected_start": start_time,
            "selected_duration": duration if duration > 0 else (total_duration - start_time)
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, midi_file, track_selection, start_time=0.0, duration=0.0):
        midi_path = folder_paths.get_full_path("midi_files", midi_file)
        if not midi_path or not os.path.isfile(midi_path):
            return f"MIDI file not found: {midi_file}"
        
        # Check if the file has a .mid or .midi extension
        if not midi_file.lower().endswith(('.mid', '.midi')):
            return f"Invalid file type. Expected .mid or .midi file, got: {midi_file}"
        
        if start_time < 0:
            return f"Start time must be non-negative, got: {start_time}"
        
        if duration < 0:
            return f"Duration must be non-negative, got: {duration}"
        
        return True

# Server routes for MIDI file handling
routes = PromptServer.instance.routes

@routes.post('/get_track_notes')
async def get_track_notes(request):
    data = await request.json()
    midi_file = data.get('midi_file')
    start_time = data.get('start_time', 0)
    duration = data.get('duration', 0)

    if not midi_file:
        return web.json_response({"error": "Missing required parameters"}, status=400)

    midi_path = folder_paths.get_full_path("midi_files", midi_file)
    if not midi_path or not os.path.exists(midi_path):
        return web.json_response({"error": "MIDI file not found"}, status=404)

    # Analyze MIDI file with time filtering
    analysis = MIDILoader.analyze_midi(midi_path, start_time, duration)
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
    start_time = data.get('start_time', 0)
    duration = data.get('duration', 0)

    if not midi_file:
        return web.json_response({"error": "Missing required parameters"}, status=400)

    midi_path = folder_paths.get_full_path("midi_files", midi_file)
    if not midi_path or not os.path.exists(midi_path):
        return web.json_response({"error": "MIDI file not found"}, status=404)

    # Get basic MIDI analysis with time filtering
    analysis = MIDILoader.analyze_midi(midi_path, start_time, duration)
    
    # Filter notes by track if needed
    if track_selection != "all":
        track_index = track_selection.split(':')[0]
        if track_index.isdigit() and track_index in analysis['track_notes']:
            analysis['all_notes'] = analysis['track_notes'][track_index]
        else:
            analysis['all_notes'] = ""
    
    return web.json_response(analysis)

NODE_CLASS_MAPPINGS = {
    "MIDIToAudio": MIDIToAudio,
    "MIDILoader": MIDILoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MIDIToAudio": "MIDI to Audio",
    "MIDILoader": "MIDI Loader"
}