import mido
import os
from .feature_pipe import FeaturePipe
import folder_paths
from server import PromptServer
from aiohttp import web
import shutil
from .midi_feature import MIDIFeature
from .feature_extractors import FeatureExtractorBase

class MIDILoadAndExtract(FeatureExtractorBase):
    @classmethod
    def feature_type(cls) -> type[MIDIFeature]:
        return MIDIFeature
    

    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "midi_file": (folder_paths.get_filename_list("midi_files"),),
                "track_selection": (["all"],),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
                "video_frames": ("IMAGE",),
                "chord_only": ("BOOLEAN", {"default": False}),
                "notes":  ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MIDI", "FEATURE", "FEATURE_PIPE")
    FUNCTION = "process_midi"
    CATEGORY = "RyanOnTheInside/Audio"

    def process_midi(self, midi_file, track_selection, notes, extraction_method, frame_rate, video_frames, chord_only=False):
        try:
            midi_path = folder_paths.get_full_path("midi_files", midi_file)
            if not midi_path or not os.path.exists(midi_path):
                raise FileNotFoundError(f"MIDI file not found: {midi_file}")

            midi_data = mido.MidiFile(midi_path)
            
            selected_notes = [int(n.strip()) for n in notes.split(',') if n.strip().isdigit()]
            feature_pipe = FeaturePipe(frame_rate, video_frames)
            
            # Convert friendly attribute name to internal attribute name
            internal_attribute = MIDIFeature.get_attribute_value(extraction_method)
            
            feature = MIDIFeature(f"midi_{internal_attribute}", midi_data, internal_attribute, 
                                  feature_pipe.frame_rate, feature_pipe.frame_count, 
                                  notes=selected_notes, chord_only=chord_only)
            
            feature.extract()

            return (midi_data, feature, feature_pipe)

        except Exception as e:
            # error_msg = f"Error in MIDILoadAndExtract.process_midi: {type(e).__name__}: {str(e)}\n"
            # error_msg += traceback.format_exc()
            # print(error_msg)
            raise RuntimeError(f"Error processing MIDI file: {type(e).__name__}: {str(e)}")

    @classmethod
    def analyze_midi(cls, midi_path):
        midi_data = mido.MidiFile(midi_path)
        tracks = ["all"]
        all_notes = set()
        track_notes = {}
        for i, track in enumerate(midi_data.tracks):
            track_notes[str(i)] = set()
            for msg in track:
                if msg.type == 'note_on':
                    track_notes[str(i)].add(msg.note)
                    all_notes.add(msg.note)
            if len(track_notes[str(i)]) == 0:
                tracks.append(f"{i}: (Empty)")
            else:
                tracks.append(f"{i}: {track.name or f'Track {i}'}")
        
        return {
            "tracks": tracks,
            "all_notes": ",".join(map(str, sorted(set(all_notes)))),
            "track_notes": {k: ",".join(map(str, sorted(v))) for k, v in track_notes.items()}
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, midi_file, track_selection, notes, extraction_method, frame_rate, video_frames):
        midi_path = folder_paths.get_full_path("midi_files", midi_file)
        if not midi_path or not os.path.isfile(midi_path):
            return f"MIDI file not found: {midi_file}"
        
        # Check if the file has a .mid or .midi extension
        if not midi_file.lower().endswith(('.mid', '.midi')):
            return f"Invalid file type. Expected .mid or .midi file, got: {midi_file}"
        
        if notes != "all":
            try:
                note_list = [int(n.strip()) for n in notes.split(',') if n.strip()]
                if not all(0 <= n <= 127 for n in note_list):
                    return "Invalid note value. All notes must be between 0 and 127."
            except ValueError:
                return "Invalid notes format. Please provide comma-separated integers or 'all'."
        
        return True


routes = PromptServer.instance.routes
@PromptServer.instance.routes.post('/get_track_notes')
async def get_track_notes(request):
    data = await request.json()
    midi_file = data.get('midi_file')

    if not midi_file:
        return web.json_response({"error": "Missing required parameters"}, status=400)

    midi_path = folder_paths.get_full_path("midi_files", midi_file)
    if not midi_path or not os.path.exists(midi_path):
        return web.json_response({"error": "MIDI file not found"}, status=404)

    analysis = MIDILoadAndExtract.analyze_midi(midi_path)
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
        analysis = MIDILoadAndExtract.analyze_midi(midi_path)

        return web.json_response({
            "status": "success",
            "uploaded_file": safe_filename,
            "midi_files": midi_files,
            "analysis": analysis
        })
    else:
        return web.json_response({"status": "error", "message": "No file uploaded"}, status=400)
    
@PromptServer.instance.routes.post('/refresh_midi_data')
async def refresh_midi_data(request):
    data = await request.json()
    midi_file = data.get('midi_file')
    track_selection = data.get('track_selection')

    if not midi_file:
        return web.json_response({"error": "Missing required parameters"}, status=400)

    midi_path = folder_paths.get_full_path("midi_files", midi_file)
    if not midi_path or not os.path.exists(midi_path):
        return web.json_response({"error": "MIDI file not found"}, status=404)

    analysis = MIDILoadAndExtract.analyze_midi(midi_path)
    
    # Filter notes based on track selection
    if track_selection != "all":
        track_index = track_selection.split(':')[0]
        analysis['all_notes'] = analysis['track_notes'].get(track_index, "")

    return web.json_response(analysis)