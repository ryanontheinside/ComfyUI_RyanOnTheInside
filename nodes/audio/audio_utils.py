import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn.functional as nnf
import librosa
import numpy as np

#utility functions
def pad_audio(waveform, pad_left, pad_right, pad_mode):
    return nnf.pad(waveform, (pad_left, pad_right), mode=pad_mode)

def normalize_volume(waveform, target_level):
    rms = torch.sqrt(torch.mean(waveform ** 2))
    current_db = 20 * torch.log10(rms + 1e-6)
    gain_db = target_level - current_db.item()
    gain = 10 ** (gain_db / 20)
    return waveform * gain

def resample_audio(waveform, orig_sample_rate, new_sample_rate):
    resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
    return resampler(waveform)

def merge_channels(waveform1, waveform2):
    min_length = min(waveform1.shape[-1], waveform2.shape[-1])
    waveform1 = waveform1[..., :min_length]
    waveform2 = waveform2[..., :min_length]
    return torch.cat([waveform1, waveform2], dim=0)

def split_channels(waveform):
    return [waveform[i:i+1, :] for i in range(waveform.shape[0])]

def concatenate_audio(waveform1, waveform2):
    return torch.cat([waveform1, waveform2], dim=-1)

def combine_audio(waveform1, waveform2, weight1=1.0, weight2=1.0):
    max_length = max(waveform1.shape[-1], waveform2.shape[-1])
    
    if waveform1.shape[-1] < max_length:
        waveform1 = nnf.pad(waveform1, (0, max_length - waveform1.shape[-1]))
    
    if waveform2.shape[-1] < max_length:
        waveform2 = nnf.pad(waveform2, (0, max_length - waveform2.shape[-1]))
    
    combined_waveform = (waveform1 * weight1) + (waveform2 * weight2)
    
    max_amplitude = torch.max(torch.abs(combined_waveform))
    if max_amplitude > 1.0:
        combined_waveform = combined_waveform / max_amplitude
    
    return combined_waveform

def pitch_shift_audio(waveform, sample_rate, n_steps):
    return F.pitch_shift(waveform, sample_rate, n_steps=n_steps)

def fade_audio(waveform, sample_rate, fade_in_duration, fade_out_duration, shape):
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)
    fader = T.Fade(
        fade_in_len=fade_in_samples,
        fade_out_len=fade_out_samples,
        fade_shape=shape
    )
    return fader(waveform)

def apply_gain(waveform, gain_db):
    gain_factor = 10 ** (gain_db / 20)
    return waveform * gain_factor

def dither_audio(waveform, bit_depth, noise_shaping):
    density_function = "TPDF" if noise_shaping == "triangular" else "RPDF"
    noise_shaping_flag = noise_shaping == "triangular"
    
    dithered_waveform = F.dither(
        waveform,
        density_function=density_function,
        noise_shaping=noise_shaping_flag
    )

    max_val = 2 ** (bit_depth - 1) - 1
    quantized_waveform = torch.clamp(dithered_waveform, -1.0, 1.0)
    return torch.round(quantized_waveform * max_val) / max_val

def time_stretch(waveform, rate):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 3:
        raise ValueError("Input waveform has too many dimensions")
    
    waveform_np = waveform.squeeze(0).numpy()
    
    stretched_channels = []
    for channel in waveform_np:
        stretched = librosa.effects.time_stretch(channel, rate=rate)
        stretched_channels.append(stretched)
    
    return torch.from_numpy(np.stack(stretched_channels)).unsqueeze(0)

