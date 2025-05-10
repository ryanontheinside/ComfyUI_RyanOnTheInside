from .audio_nodes import AudioNodeBase
from .audio_utils import (
    pad_audio,
    normalize_volume,
    resample_audio,
    merge_channels,
    split_channels,
    concatenate_audio,
    combine_audio,
    dither_audio,
)
import torch
from ...tooltips import apply_tooltips
import librosa
import numpy as np

class AudioUtility(AudioNodeBase):
    def __init__(self):
        super().__init__()

    CATEGORY = "RyanOnTheInside/Audio/Utility"

@apply_tooltips
class AudioPad(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pad_left": ("INT", {"default": 0, "min": 0, "max": 44100, "step": 1}),
                "pad_right": ("INT", {"default": 0, "min": 0, "max": 44100, "step": 1}),
                "pad_mode": (["constant", "reflect", "replicate", "circular"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "pad_audio_node"

    def pad_audio_node(self, audio, pad_left, pad_right, pad_mode):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        padded_waveform = pad_audio(waveform, pad_left, pad_right, pad_mode)
        return ({"waveform": padded_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class AudioVolumeNormalization(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_level": ("FLOAT", {"default": -10.0, "min": -60.0, "max": 0.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "normalize_volume_node"

    def normalize_volume_node(self, audio, target_level):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        normalized_waveform = normalize_volume(waveform, target_level)
        return ({"waveform": normalized_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class AudioResample(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "new_sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 100}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "resample_audio_node"

    def resample_audio_node(self, audio, new_sample_rate):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        resampled_waveform = resample_audio(waveform, sample_rate, new_sample_rate)
        return ({"waveform": resampled_waveform, "sample_rate": new_sample_rate},)

@apply_tooltips
class AudioChannelMerge(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("LIST[AUDIO]",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "merge_channels_node"

    def merge_channels_node(self, audio_list):
        waveform_list = [audio['waveform'] for audio in audio_list]
        sample_rate = audio_list[0]['sample_rate']

        # Check that all sample rates are the same
        for audio in audio_list:
            if audio['sample_rate'] != sample_rate:
                raise ValueError("Sample rates must match for channel merging")

        merged_waveform = merge_channels(waveform_list)
        return ({"waveform": merged_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class AudioChannelSplit(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("LIST[AUDIO]",)
    FUNCTION = "split_channels_node"

    def split_channels_node(self, audio):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        channel_waveforms = split_channels(waveform)
        audio_list = [{"waveform": w, "sample_rate": sample_rate} for w in channel_waveforms]
        return (audio_list,)

@apply_tooltips
class Audio_Concatenate(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concatenate_audio_node"

    def concatenate_audio_node(self, audio1, audio2):
        if audio1['sample_rate'] != audio2['sample_rate']:
            raise ValueError("Both audio inputs must have the same sample rate")

        sample_rate = audio1['sample_rate']
        concatenated_waveform = concatenate_audio(audio1['waveform'], audio2['waveform'])
        return ({"waveform": concatenated_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class Audio_Combine(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
                "weight1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "combine_audio_node"

    def combine_audio_node(self, audio1, audio2, weight1=0.5, weight2=0.5):
        if audio1['sample_rate'] != audio2['sample_rate']:
            raise ValueError("Both audio inputs must have the same sample rate")

        sample_rate = audio1['sample_rate']
        combined_waveform = combine_audio(audio1['waveform'], audio2['waveform'], weight1, weight2)
        return ({"waveform": combined_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class AudioSubtract(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "audio2": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "subtract_audio_node"

    def subtract_audio_node(self, audio1, audio2):
        if audio1['sample_rate'] != audio2['sample_rate']:
            raise ValueError("Both audio inputs must have the same sample rate")

        sample_rate = audio1['sample_rate']
        waveform1 = audio1['waveform']
        waveform2 = audio2['waveform']

        # Ensure both waveforms have the same length
        if waveform1.shape[1] != waveform2.shape[1]:
            max_length = max(waveform1.shape[1], waveform2.shape[1])
            waveform1 = torch.nn.functional.pad(waveform1, (0, max_length - waveform1.shape[1]))
            waveform2 = torch.nn.functional.pad(waveform2, (0, max_length - waveform2.shape[1]))

        subtracted_waveform = waveform1 - waveform2
        return ({"waveform": subtracted_waveform, "sample_rate": sample_rate},)

#TODO: TOO SLOW
@apply_tooltips
class AudioInfo(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "total_frames", "frames_per_beat", "frames_per_bar", "frames_per_quarter", "frames_per_eighth",
        "audio_duration", "beats_per_second", "detected_bpm",
        "sample_rate", "num_channels", "num_samples",
        "max_amplitude", "mean_amplitude", "rms_amplitude", "bit_depth"
    )
    FUNCTION = "get_audio_info"

    def get_audio_info(self, audio, frame_rate):
        # Get basic audio info
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        # Calculate original audio info first
        num_channels = waveform.shape[1] if waveform.dim() > 2 else 1
        num_samples = waveform.shape[-1]
        audio_duration = num_samples / sample_rate
        
        # Calculate total frames
        total_frames = int(audio_duration * frame_rate)
        
        # Detect BPM using librosa
        audio_mono = waveform.squeeze(0).mean(axis=0).cpu().numpy()
        tempo, _ = librosa.beat.beat_track(y=audio_mono, sr=sample_rate)
        beats_per_second = tempo / 60.0
        
        # Calculate frames per beat and musical divisions
        frames_per_beat = int(frame_rate / beats_per_second)
        frames_per_bar = frames_per_beat * 4  # Assuming 4/4 time signature
        frames_per_quarter = frames_per_beat
        frames_per_eighth = frames_per_beat // 2
        
        # Calculate amplitude statistics
        max_amplitude = float(torch.max(torch.abs(waveform)))
        mean_amplitude = float(torch.mean(torch.abs(waveform)))
        rms_amplitude = float(torch.sqrt(torch.mean(waveform ** 2)))
        
        # Get bit depth from dtype
        bit_depth = str(waveform.dtype)
        
        return (
            total_frames,
            frames_per_beat,
            frames_per_bar,
            frames_per_quarter,
            frames_per_eighth,
            audio_duration,
            beats_per_second,
            tempo,  # detected_bpm
            sample_rate,
            num_channels,
            num_samples,
            max_amplitude,
            mean_amplitude,
            rms_amplitude,
            bit_depth
        )

@apply_tooltips
class AudioDither(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "bit_depth": ("INT", {"default": 16, "min": 8, "max": 32, "step": 1}),
                "noise_shaping": (["none", "triangular"],),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "dither_audio_node"

    def dither_audio_node(self, audio, bit_depth, noise_shaping):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        quantized_waveform = dither_audio(waveform, bit_depth, noise_shaping)
        return ({"waveform": quantized_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class AudioTrim(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
                "end_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "trim_audio"

    def trim_audio(self, audio, start_time, end_time):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        
        # Convert time to samples
        start_sample = int(start_time * sample_rate)
        
        # If end_time is 0 or less, use the full audio duration
        if end_time <= 0:
            end_sample = waveform.shape[-1]
        else:
            end_sample = int(end_time * sample_rate)
            
        # Ensure end_sample is not out of bounds
        end_sample = min(end_sample, waveform.shape[-1])
        
        # Ensure start_sample is not greater than end_sample
        start_sample = min(start_sample, end_sample)
        
        # Trim the audio
        trimmed_waveform = waveform[..., start_sample:end_sample]
        
        return ({"waveform": trimmed_waveform, "sample_rate": sample_rate},)

@apply_tooltips
class Knob(AudioUtility):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "knob": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "knob"}),
                "other_knob": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "enhance_audio"

    def enhance_audio(self, audio, knob, other_knob):
        waveform, sample_rate = audio['waveform'], audio['sample_rate']
        
        # Make a copy of the input waveform
        enhanced = waveform.clone()
        
        if knob > 0:
            # Apply effects with intensity based on enhancement and sickness levels
            intensity = knob * (1.0 + other_knob)
            
            # 1. Multiband compression - separately process low, mid, and high frequencies
            # Convert to frequency domain for multiband processing
            enhanced_np = enhanced.squeeze(0).cpu().numpy()
            
            if enhanced_np.ndim > 1 and enhanced_np.shape[0] > 1:  # Stereo processing
                num_channels = enhanced_np.shape[0]
                processed_channels = []
                
                for ch in range(num_channels):
                    channel_data = enhanced_np[ch]
                    
                    # Split into frequency bands
                    stft = librosa.stft(channel_data)
                    freq_bins = stft.shape[0]
                    
                    # Define frequency bands (low, mid, high)
                    low_band = int(freq_bins * 0.15)  # 0-15% of frequency range
                    mid_band = int(freq_bins * 0.6)   # 15-60% of frequency range
                    
                    # Apply different compression to each band
                    # Low frequencies - heavy compression for tight bass
                    low_comp = 0.3 + (intensity * 0.5)  # 0.3-0.8
                    stft[:low_band] *= (1.0 + torch.tanh(torch.tensor(abs(stft[:low_band])) * low_comp).numpy())
                    
                    # Mid frequencies - moderate compression for vocal clarity
                    mid_comp = 0.2 + (intensity * 0.3)  # 0.2-0.5
                    stft[low_band:mid_band] *= (1.0 + (intensity * 0.4)) 
                    
                    # High frequencies - excitement and air
                    high_boost = 0.3 + (intensity * 0.7)  # 0.3-1.0
                    stft[mid_band:] *= (1.0 + high_boost)
                    
                    # Convert back to time domain
                    processed_channel = librosa.istft(stft, length=len(channel_data))
                    processed_channels.append(processed_channel)
                
                enhanced_np = np.array(processed_channels)
            else:
                # Mono processing
                stft = librosa.stft(enhanced_np)
                freq_bins = stft.shape[0]
                
                # Define frequency bands
                low_band = int(freq_bins * 0.15)
                mid_band = int(freq_bins * 0.6)
                
                # Apply band-specific processing
                low_comp = 0.3 + (intensity * 0.5)
                stft[:low_band] *= (1.0 + np.tanh(np.abs(stft[:low_band]) * low_comp))
                stft[low_band:mid_band] *= (1.0 + (intensity * 0.4))
                stft[mid_band:] *= (1.0 + (0.3 + (intensity * 0.7)))
                
                # Convert back
                enhanced_np = librosa.istft(stft, length=len(enhanced_np))
                enhanced_np = np.expand_dims(enhanced_np, axis=0)
                
            # Convert back to torch tensor
            enhanced = torch.from_numpy(enhanced_np).unsqueeze(0).to(waveform.device)
            
            # 2. Stereo enhancement (if stereo audio)
            if enhanced.shape[1] >= 2:
                # Haas effect - subtle delay between channels creates wider stereo image
                delay_samples = int(sample_rate * 0.01 * intensity)  # 0-10ms delay
                if delay_samples > 0:
                    left = enhanced[:, 0:1, :]
                    right = enhanced[:, 1:2, :]
                    
                    # Apply delay to right channel
                    delayed_right = torch.zeros_like(right)
                    delayed_right[:, :, delay_samples:] = right[:, :, :-delay_samples]
                    
                    # Mix original and delayed signal
                    mix_ratio = 0.7 + (intensity * 0.3)  # 0.7-1.0
                    enhanced_right = right * (1 - mix_ratio) + delayed_right * mix_ratio
                    
                    # Recombine channels
                    enhanced = torch.cat([left, enhanced_right], dim=1)
                
                # Increase stereo width
                mid = (enhanced[:, 0:1, :] + enhanced[:, 1:2, :]) * 0.5
                side = (enhanced[:, 0:1, :] - enhanced[:, 1:2, :]) * 0.5
                
                # Boost side signal for wider stereo image
                side_boost = 1.0 + (intensity * other_knob * 1.5)  # 1.0-2.5x
                side = side * side_boost
                
                # Recombine mid and side
                left_new = mid + side
                right_new = mid - side
                enhanced = torch.cat([left_new, right_new], dim=1)
            
            # 3. Add saturation/distortion based on sickness level
            if other_knob > 0.3:
                drive = 1.0 + (other_knob * 5.0)  # 1.0-6.0
                enhanced = torch.tanh(enhanced * drive) / torch.tanh(torch.tensor(drive))
            
            # 4. Apply "exciter" effect - add synthetic harmonics
            if other_knob > 0.2:
                # Generate harmonics through waveshaping
                harmonics = torch.sin(enhanced * (3.14159 * (1.0 + other_knob * 3.0)))
                enhanced = enhanced * (0.7 + (other_knob * 0.1)) + harmonics * (0.2 + (other_knob * 0.3))
            
            # 5. Bass boost for extra sickness
            if other_knob > 0.1 and sample_rate >= 44100:
                enhanced_np = enhanced.squeeze(0).cpu().numpy()
                
                # Handle stereo or mono audio appropriately
                if enhanced_np.ndim > 1 and enhanced_np.shape[0] > 1:  # Stereo
                    processed_channels = []
                    
                    for ch in range(enhanced_np.shape[0]):
                        # Process each channel independently
                        channel_data = enhanced_np[ch]
                        stft = librosa.stft(channel_data)
                        
                        # Create bass boost filter for this channel
                        freq_bins = stft.shape[0]
                        bass_gain = 1.0 + (other_knob * intensity * 2.0)
                        bass_shelf = np.ones(freq_bins)
                        bass_end = int(freq_bins * 0.1)
                        bass_shelf[:bass_end] = bass_gain
                        
                        # Reshape for broadcasting and apply
                        bass_shelf = bass_shelf.reshape(-1, 1)
                        stft = stft * bass_shelf
                        
                        # Convert back to time domain
                        processed_channel = librosa.istft(stft, length=len(channel_data))
                        processed_channels.append(processed_channel)
                    
                    # Combine channels back
                    enhanced_np = np.array(processed_channels)
                else:
                    # Mono processing
                    stft = librosa.stft(enhanced_np.squeeze())
                    
                    # Create bass boost filter
                    freq_bins = stft.shape[0]
                    bass_gain = 1.0 + (other_knob * intensity * 2.0)
                    bass_shelf = np.ones(freq_bins)
                    bass_end = int(freq_bins * 0.1)
                    bass_shelf[:bass_end] = bass_gain
                    
                    # Reshape for broadcasting and apply
                    bass_shelf = bass_shelf.reshape(-1, 1)
                    stft = stft * bass_shelf
                    
                    # Convert back to time domain
                    enhanced_np = librosa.istft(stft, length=len(enhanced_np.squeeze()))
                    enhanced_np = np.expand_dims(enhanced_np, axis=0)
                
                # Convert back to torch tensor
                enhanced = torch.from_numpy(enhanced_np).unsqueeze(0).to(waveform.device)
            
            # 6. Normalize levels with a target based on enhancement
            target_level = -6.0 - ((1.0 - intensity) * 6.0)  # -12dB to -6dB
            enhanced = normalize_volume(enhanced, target_level)
            
            # Final limiting to prevent clipping
            max_val = torch.max(torch.abs(enhanced))
            if max_val > 0.99:
                enhanced = enhanced / max_val * 0.99
            
            # Add subtle pumping effect for EDM-style sickness if sickness > 0.7
            if other_knob > 0.7:
                # Create a pumping/sidechain effect
                pump_rate = int(sample_rate * (60/128) * 0.25)  # Assuming 128 BPM, quarter note
                pump_envelope = torch.ones_like(enhanced)
                
                for i in range(0, enhanced.shape[-1], pump_rate):
                    end_idx = min(i + pump_rate, enhanced.shape[-1])
                    pump_len = end_idx - i
                    # Create attack-release envelope
                    env = torch.linspace(0.5, 1.0, pump_len).to(enhanced.device)
                    for c in range(enhanced.shape[1]):
                        pump_envelope[:, c, i:end_idx] = env.unsqueeze(0)
                
                # Apply pumping
                enhanced = enhanced * pump_envelope
        
        return ({"waveform": enhanced, "sample_rate": sample_rate},)