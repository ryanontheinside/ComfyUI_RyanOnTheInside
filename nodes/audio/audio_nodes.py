import torch
import numpy as np
from scipy.ndimage import zoom
from .audio_processor_legacy import AudioVisualizer, PygameAudioVisualizer
from scipy import signal
import comfy.model_management as mm
import folder_paths
import os
from ... import RyanOnTheInside
from ..flex.feature_pipe import FeaturePipe
import hashlib
import torchaudio
from ...tooltips import apply_tooltips


class AudioNodeBase(RyanOnTheInside):
    CATEGORY= "RyanOnTheInside/Audio"
    @staticmethod
    def create_empty_tensor(audio, frame_rate, height, width, channels=None):
        audio_duration = audio['waveform'].shape[-1] / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)
        if channels is None:
            return torch.zeros((num_frames, height, width), dtype=torch.float32)
        else:
            return torch.zeros((num_frames, height, width, channels), dtype=torch.float32)

@apply_tooltips
class DownloadOpenUnmixModel(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["umxl", "umxhq"], {"default": "umxl"}),
            }
        }

    RETURN_TYPES = ("OPEN_UNMIX_MODEL",)
    FUNCTION = "download_and_load_model"
    CATEGORY = "RyanOnTheInside/Audio/AudioSeparation"

    def download_and_load_model(self, model_name):
        device = mm.get_torch_device()
        download_path = os.path.join(folder_paths.models_dir, "openunmix")
        os.makedirs(download_path, exist_ok=True)

        model_file = f"{model_name}.pth"
        model_path = os.path.join(download_path, model_file)

        if not os.path.exists(model_path):
            print(f"Downloading {model_name} model...")
            separator = torch.hub.load('sigsep/open-unmix-pytorch', model_name, device='cpu')
            torch.save(separator.state_dict(), model_path)
            print(f"Model saved to: {model_path}")
        else:
            print(f"Loading model from: {model_path}")
            separator = torch.hub.load('sigsep/open-unmix-pytorch', model_name, device='cpu')
            separator.load_state_dict(torch.load(model_path, map_location='cpu'))

        separator = separator.to(device)
        separator.eval()

        return (separator,)



@apply_tooltips
class AudioSeparatorSimple(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OPEN_UNMIX_MODEL",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("audio", "drums_audio", "vocals_audio", "bass_audio", "other_audio")
    FUNCTION = "process_audio"
    CATEGORY = "RyanOnTheInside/Audio/AudioSeparation"

    def process_audio(self, model, audio):
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']

        if waveform.dim() == 3:
            waveform = waveform.squeeze(0) 
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension if mono
        if waveform.shape[0] != 2:
            waveform = waveform.repeat(2, 1)  # Duplicate mono to stereo if necessary
            
        waveform = waveform.unsqueeze(0)

        # Move to appropriate device
        device = next(model.parameters()).device
        waveform = waveform.to(device)

        # Perform separation
        estimates = model(waveform)

        # Create isolated audio objects
        isolated_audio = {}
        target_indices = {'drums': 1, 'vocals': 0, 'bass': 2, 'other': 3}
        for target, index in target_indices.items():
            target_waveform = estimates[:, index, :, :]
            isolated_audio[target] = {
                'waveform': target_waveform.cpu(),
                'sample_rate': sample_rate
            }

        return (
            audio,
            isolated_audio['drums'],
            isolated_audio['vocals'],
            isolated_audio['bass'],
            isolated_audio['other']
        )

@apply_tooltips
class AudioFilter(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filters": ("FREQUENCY_FILTER",)
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "apply_filters"
    DESCRIPTION = """Applies frequency filters to audio:
- `audio`: Input audio to be filtered
- `filters`: Frequency filters to be applied (FREQUENCY_FILTER type)"""
    CATEGORY = "RyanOnTheInside/Audio/Filters"
    def apply_filters(self, audio, filters):
        audio_np = audio['waveform'].cpu().numpy().squeeze(0)
        sample_rate = audio['sample_rate']

        filtered_channels = []
        for channel in audio_np:
            filtered_channel = channel
            for filter_params in filters:
                filtered_channel = self.apply_single_filter(filtered_channel, filter_params, sample_rate)
            filtered_channels.append(filtered_channel)

        filtered_audio = np.stack(filtered_channels)

        max_val = np.max(np.abs(filtered_audio))
        if max_val > 1.0:
            filtered_audio = filtered_audio / max_val

        filtered_tensor = torch.from_numpy(filtered_audio).unsqueeze(0).float()

        return ({"waveform": filtered_tensor, "sample_rate": sample_rate},)

    def apply_single_filter(self, audio, filter_params, sample_rate):
        filter_type = filter_params['type']
        order = filter_params['order']
        cutoff = filter_params['cutoff']

        if filter_type == 'lowpass':
            b, a = signal.butter(order, cutoff, btype='low', fs=sample_rate)
        elif filter_type == 'highpass':
            b, a = signal.butter(order, cutoff, btype='high', fs=sample_rate)
        elif filter_type == 'bandpass':
            low = cutoff - 50
            high = cutoff + 50
            b, a = signal.butter(order, [low, high], btype='band', fs=sample_rate)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        return signal.lfilter(b, a, audio)

# class DownloadCREPEModel(AudioNodeBase):
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "model_name": (["tiny", "small", "medium", "large", "full"], {"default": "medium"}),
#             }
#         }

#     RETURN_TYPES = ("CREPE_MODEL",)
#     FUNCTION = "download_and_load_model"
#     CATEGORY = "RyanOnTheInside/Audio"

#     def download_and_load_model(self, model_name):
#         try:
#             import crepe
#         except ImportError:
#             raise ImportError("""To use this node please 
#                               pip install crepe tensorflow
#                               :)
#                               """)

#         download_path = os.path.join(folder_paths.models_dir, "crepe")
#         os.makedirs(download_path, exist_ok=True)

#         model_file = f"crepe_{model_name}.json"
#         model_path = os.path.join(download_path, model_file)

#         if not os.path.exists(model_path):
#             print(f"Downloading CREPE {model_name} model...")
#             model = crepe.core.build_and_load_model(model_name)
#             print(f"Model downloaded and loaded.")
#         else:
#             print(f"Loading model from: {model_path}")
#             model = crepe.core.build_and_load_model(model_name)

#         return (model,)

@apply_tooltips
class FrequencyFilterPreset(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([
                    "isolate_kick_drum",
                    "isolate_snare_drum",
                    "isolate_hi_hat",
                    "isolate_bass",
                    "isolate_vocals",
                    "remove_rumble",
                    "brighten_mix",
                    "remove_hiss"
                ],),
            },
            "optional": {
                "previous_filter": ("FREQUENCY_FILTER",),
            },
        }

    RETURN_TYPES = ("FREQUENCY_FILTER",)
    FUNCTION = "create_preset_filter_chain"
    CATEGORY = "RyanOnTheInside/Audio/Filters"

    def create_preset_filter_chain(self, preset, previous_filter=None):
        new_filters = self.get_preset_filters(preset)

        if previous_filter:
            if isinstance(previous_filter, list):
                return (previous_filter + new_filters,)
            else:
                return ([previous_filter] + new_filters,)
        else:
            return (new_filters,)

    def get_preset_filters(self, preset):
        if preset == "isolate_kick_drum":
            return [
                {"type": "highpass", "order": 2, "cutoff": 30},
                {"type": "lowpass", "order": 4, "cutoff": 200}
            ]
        elif preset == "isolate_snare_drum":
            return [
                {"type": "bandpass", "order": 4, "cutoff": 200},
                {"type": "bandpass", "order": 4, "cutoff": 5000}
            ]
        elif preset == "isolate_hi_hat":
            return [
                {"type": "highpass", "order": 4, "cutoff": 8000}
            ]
        elif preset == "isolate_bass":
            return [
                {"type": "lowpass", "order": 4, "cutoff": 250}
            ]
        elif preset == "isolate_vocals":
            return [
                {"type": "bandpass", "order": 2, "cutoff": 300},
                {"type": "bandpass", "order": 2, "cutoff": 4000}
            ]
        elif preset == "remove_rumble":
            return [
                {"type": "highpass", "order": 4, "cutoff": 40}
            ]
        elif preset == "brighten_mix":
            return [
                {"type": "highpass", "order": 2, "cutoff": 6000}
            ]
        elif preset == "remove_hiss":
            return [
                {"type": "lowpass", "order": 6, "cutoff": 7500}
            ]
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
@apply_tooltips
class FrequencyFilterCustom(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filter_type": (["lowpass", "highpass", "bandpass"],),
                "order": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "cutoff": ("FLOAT", {"default": 1000, "min": 20, "max": 20000, "step": 1}),
            },
            "optional": {
                "previous_filter": ("FREQUENCY_FILTER",),
            },
        }

    RETURN_TYPES = ("FREQUENCY_FILTER",)
    FUNCTION = "create_filter"
    CATEGORY = "RyanOnTheInside/Audio/Filters"

    def create_filter(self, filter_type, order, cutoff, previous_filter=None):
        filter_params = {
            "type": filter_type,
            "order": order,
            "cutoff": cutoff,
        }

        if previous_filter:
            if isinstance(previous_filter, list):
                previous_filter.append(filter_params)
                return (previous_filter,)
            else:
                return ([previous_filter, filter_params],)
        else:
            return ([filter_params],)

@apply_tooltips
class FrequencyRange(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_cutoff": ("FLOAT", {"default": 20, "min": 20, "max": 20000, "step": 1}),
                "high_cutoff": ("FLOAT", {"default": 20000, "min": 20, "max": 20000, "step": 1}),
                "order": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "previous_range": ("FREQUENCY_RANGE",),
            },
        }

    RETURN_TYPES = ("FREQUENCY_RANGE",)
    FUNCTION = "create_frequency_range"
    CATEGORY = "RyanOnTheInside/Audio/FrequencyBands"

    def create_frequency_range(self, low_cutoff, high_cutoff, order, previous_range=None):
        range_params = {
            "type": "bandpass",
            "order": order,
            "low_cutoff": low_cutoff,
            "high_cutoff": high_cutoff,
        }

        if previous_range:
            if isinstance(previous_range, list):
                previous_range.append(range_params)
                return (previous_range,)
            else:
                return ([previous_range, range_params],)
        else:
            return ([range_params],)
        
@apply_tooltips
class AudioFeatureVisualizer(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
                "visualization_type": ([
                    "waveform", 
                    "spectrogram", 
                    "mfcc", 
                    "chroma", 
                    "tonnetz", 
                    "spectral_centroid"
                    ],),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
                "x_axis": ([
                    "time", 
                    "mel", 
                    "log", 
                    "frames", 
                    "off", 
                    "chroma", 
                    None
                    ],),
                "y_axis": ([
                    "linear", 
                    "log", 
                    "mel", 
                    "chroma", 
                    "tonnetz", 
                    "frames", 
                    "off", 
                    None
                    ],),
                "cmap": ([
                    "magma", 
                    "inferno", 
                    "plasma", 
                    "viridis", 
                    "gray", 
                    "coolwarm", 
                    "cividis", 
                    "jet", 
                    "hot", 
                    "cubehelix", 
                    "Blues"
                    ],),
                "visualizer": ([
                    "pygame",
                    "matplotlib"
                    ],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_audio_feature"
    CATEGORY = "audio"
    CATEGORY = "RyanOnTheInside/Audio/Filters"

    def visualize_audio_feature(self, audio, video_frames, visualization_type, frame_rate, x_axis, y_axis, cmap, visualizer):
        num_frames, height, width, _ = video_frames.shape
        if visualizer == "pygame":  
            visualizer = PygameAudioVisualizer(audio, num_frames, height, width, frame_rate)
        elif visualizer == "matplotlib":
            visualizer = AudioVisualizer(audio, num_frames, height, width, frame_rate, x_axis, y_axis, cmap)
        
        if visualization_type == "waveform":
            mask = visualizer.create_waveform()
        elif visualization_type == "spectrogram":
            mask = visualizer.create_spectrogram()
        elif visualization_type == "mfcc":
            mask = visualizer.create_mfcc()
        elif visualization_type == "chroma":
            mask = visualizer.create_chroma()
        elif visualization_type == "tonnetz":
            mask = visualizer.create_tonnetz()
        elif visualization_type == "spectral_centroid":
            mask = visualizer.create_spectral_centroid()
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")

        return (mask,)  

@apply_tooltips
class EmptyImageFromAudio(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("empty_image", "frame_count")
    FUNCTION = "create_empty_image"
    CATEGORY = "RyanOnTheInside/Audio/Utility"

    def create_empty_image(self, audio, frame_rate, height, width):
        empty_image = self.create_empty_tensor(audio, frame_rate, height, width, channels=3)
        frame_count = empty_image.shape[0]
        return (empty_image, frame_count)

@apply_tooltips
class EmptyMaskFromAudio(AudioNodeBase):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("empty_mask", "frame_count")
    FUNCTION = "create_empty_mask"
    CATEGORY = "RyanOnTheInside/Audio/Utility"

    def create_empty_mask(self, audio, frame_rate, height, width):
        empty_mask = self.create_empty_tensor(audio, frame_rate, height, width)
        frame_count = empty_mask.shape[0]
        return (empty_mask, frame_count)

# class FlexLoadAudio(AudioNodeBase):
#     @classmethod
#     def INPUT_TYPES(cls):
#         input_dir = folder_paths.get_input_directory()
#         files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
#         return {
#             "required": {
#                 "audio": (sorted(files), {"audio_upload": True}),
#                 "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
#                 "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
#                 "width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
#             }
#         }

#     RETURN_TYPES = ("AUDIO", "IMAGE", "MASK")
#     RETURN_NAMES = ("audio", "empty_image", "empty_mask")
#     FUNCTION = "load_and_create_empty"
#     CATEGORY = "RyanOnTheInside/Audio/Utility"

#     def load_and_create_empty(self, audio, frame_rate, height, width):
#         audio_path = folder_paths.get_annotated_filepath(audio)
#         waveform, sample_rate = torchaudio.load(audio_path)
#         audio_data = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate, "frame_rate": frame_rate}
        
#         # Create empty image and mask
#         audio_duration = waveform.shape[-1] / sample_rate
#         num_frames = int(audio_duration * frame_rate)
#         empty_image = torch.zeros((num_frames, height, width, 3), dtype=torch.float32)
#         empty_mask = torch.zeros((num_frames, height, width), dtype=torch.float32)
        
#         return (audio_data, empty_image, empty_mask)

#     @classmethod
#     def IS_CHANGED(cls, audio):
#         image_path = folder_paths.get_annotated_filepath(audio)
#         m = hashlib.sha256()
#         with open(image_path, 'rb') as f:
#             m.update(f.read())
#         return m.digest().hex()

#     @classmethod
#     def VALIDATE_INPUTS(cls, audio):
#         if not folder_paths.exists_annotated_filepath(audio):
#             return "Invalid audio file: {}".format(audio)
#         return True

@apply_tooltips
class EmptyImageAndMaskFromAudio(AudioNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
                "width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 1}),
                
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("empty_image", "empty_mask", "frame_count")
    FUNCTION = "create_empty_image_and_mask"
    CATEGORY = "RyanOnTheInside/Audio/Utility"

    def create_empty_image_and_mask(self, audio, frame_rate, height, width):
        empty_image = self.create_empty_tensor(audio, frame_rate, height, width, channels=3)
        empty_mask = self.create_empty_tensor(audio, frame_rate, height, width)
        frame_count = empty_image.shape[0]
        return (empty_image, empty_mask, frame_count)