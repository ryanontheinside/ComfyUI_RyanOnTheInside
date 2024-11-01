import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import librosa
import numpy as np
import torch.nn.functional as nnf

def pitch_shift(waveform, sample_rate, n_steps):
    """
    Pitch shifts the waveform by n_steps semitones.

    Args:
        waveform (Tensor): The input waveform tensor.
        sample_rate (int): The sample rate of the waveform.
        n_steps (int): Number of steps to shift the pitch.

    Returns:
        Tensor: Pitch-shifted waveform.
    """
    # Use the functional API for pitch shifting
    shifted_waveform = F.pitch_shift(waveform, sample_rate, n_steps=n_steps)
    return shifted_waveform

def fade_audio(waveform, sample_rate, fade_in_duration, fade_out_duration, shape):
    """
    Applies fade in and fade out to the waveform.

    Args:
        waveform (Tensor): The input waveform tensor.
        sample_rate (int): The sample rate of the waveform.
        fade_in_duration (float): Duration of fade-in in seconds.
        fade_out_duration (float): Duration of fade-out in seconds.
        shape (str): Shape of the fade ("linear", "exponential", "logarithmic", etc.)

    Returns:
        Tensor: Waveform with fades applied.
    """
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)
    fader = T.Fade(
        fade_in_len=fade_in_samples,
        fade_out_len=fade_out_samples,
        fade_shape=shape
    )
    faded_waveform = fader(waveform)
    return faded_waveform

def apply_gain(waveform, gain_db):
    """
    Applies gain to the waveform.

    Args:
        waveform (Tensor): The input waveform tensor.
        gain_db (float): Gain in decibels.

    Returns:
        Tensor: Waveform with gain applied.
    """
    # Calculate the gain factor
    gain_factor = 10 ** (gain_db / 20)
    amplified_waveform = waveform * gain_factor
    return amplified_waveform

def time_stretch(waveform, rate):
    """
    Stretches or compresses the waveform in time.

    Args:
        waveform (Tensor): The input waveform tensor.
        rate (float): Rate to stretch the waveform (e.g., 2.0 doubles the length).

    Returns:
        Tensor: Time-stretched waveform.
    """
    # Ensure the input is 2D (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() > 3:
        raise ValueError("Input waveform has too many dimensions")

    # Move tensor to CPU if it's on GPU
    device = waveform.device
    waveform = waveform.cpu()

    # Convert to numpy for librosa processing
    waveform_np = waveform.numpy()

    # Process each channel
    stretched_channels = []
    for channel in waveform_np:
        stretched = librosa.effects.time_stretch(channel, rate=rate)
        stretched_channels.append(stretched)

    # Stack channels and convert back to torch tensor
    stretched_waveform = torch.from_numpy(np.stack(stretched_channels))

    # Move the tensor back to the original device
    stretched_waveform = stretched_waveform.to(device)

    return stretched_waveform

def dither_audio(waveform, bit_depth, noise_shaping):
    """
    Applies dithering to the waveform.

    Args:
        waveform (Tensor): The input waveform tensor.
        bit_depth (int): Target bit depth.
        noise_shaping (str): Type of noise shaping ("none", "triangular").

    Returns:
        Tensor: Dithered and quantized waveform.
    """
    # Select density function based on noise shaping
    if noise_shaping == "triangular":
        density_function = "TPDF"
        noise_shaping_flag = True
    else:
        density_function = "RPDF"
        noise_shaping_flag = False

    # Apply dithering
    dithered_waveform = F.dither(
        waveform,
        density_function=density_function,
        noise_shaping=noise_shaping_flag
    )

    # Quantize the waveform to the specified bit depth
    max_val = 2 ** (bit_depth - 1) - 1
    quantized_waveform = torch.clamp(dithered_waveform, -1.0, 1.0)
    quantized_waveform = torch.round(quantized_waveform * max_val) / max_val

    return quantized_waveform

def pad_audio(waveform, pad_left, pad_right, pad_mode):
    """
    Pads the waveform on the left and right.

    Args:
        waveform (Tensor): The input waveform tensor.
        pad_left (int): Number of samples to pad on the left.
        pad_right (int): Number of samples to pad on the right.
        pad_mode (str): Padding mode ("constant", "reflect", "replicate", "circular").

    Returns:
        Tensor: Padded waveform.
    """
    # Use torch.nn.functional.pad for padding
    padded_waveform = nnf.pad(waveform, (pad_left, pad_right), mode=pad_mode)
    return padded_waveform

def normalize_volume(waveform, target_level):
    """
    Normalizes the waveform to a target RMS level in decibels.

    Args:
        waveform (Tensor): The input waveform tensor.
        target_level (float): Target RMS level in decibels.

    Returns:
        Tensor: Normalized waveform.
    """
    # Calculate current RMS level in dB
    rms = torch.sqrt(torch.mean(waveform ** 2))
    current_db = 20 * torch.log10(rms + 1e-6)  # Add small value to avoid log(0)

    # Calculate the required gain in dB
    gain_db = target_level - current_db.item()
    gain = 10 ** (gain_db / 20)

    # Apply gain
    normalized_waveform = waveform * gain

    return normalized_waveform

def resample_audio(waveform, orig_sample_rate, new_sample_rate):
    """
    Resamples the waveform to a new sample rate.

    Args:
        waveform (Tensor): The input waveform tensor.
        orig_sample_rate (int): Original sample rate.
        new_sample_rate (int): Desired sample rate.

    Returns:
        Tensor: Resampled waveform.
    """
    resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=new_sample_rate)
    resampled_waveform = resampler(waveform)
    return resampled_waveform

def merge_channels(waveform_list):
    """
    Merges multiple mono waveforms into a multi-channel waveform.

    Args:
        waveform_list (list of Tensor): List of waveform tensors to merge.

    Returns:
        Tensor: Merged waveform with multiple channels.
    """
    # Ensure all waveforms have the same length
    min_length = min(w.shape[-1] for w in waveform_list)
    trimmed_waveforms = [w[..., :min_length] for w in waveform_list]

    merged_waveform = torch.cat(trimmed_waveforms, dim=0)
    return merged_waveform

def split_channels(waveform):
    """
    Splits a multi-channel waveform into individual channels.

    Args:
        waveform (Tensor): The input multi-channel waveform tensor.

    Returns:
        list of Tensor: List containing individual channel waveforms.
    """
    if waveform.dim() < 2 or waveform.shape[0] < 2:
        raise ValueError("Input waveform must have at least 2 channels for splitting")

    # Split into individual channels
    channel_waveforms = [waveform[i:i+1, :] for i in range(waveform.shape[0])]
    return channel_waveforms

def concatenate_audio(waveform1, waveform2):
    """
    Concatenates two waveforms end-to-end.

    Args:
        waveform1 (Tensor): The first waveform tensor.
        waveform2 (Tensor): The second waveform tensor.

    Returns:
        Tensor: Concatenated waveform.
    """
    # Concatenate waveforms
    concatenated_waveform = torch.cat([waveform1, waveform2], dim=-1)
    return concatenated_waveform

def combine_audio(waveform1, waveform2, weight1=0.5, weight2=0.5):
    """
    Combines two waveforms by weighted addition.

    Args:
        waveform1 (Tensor): The first waveform tensor.
        waveform2 (Tensor): The second waveform tensor.
        weight1 (float): Weight for the first waveform.
        weight2 (float): Weight for the second waveform.

    Returns:
        Tensor: Combined waveform.
    """
    # Get the maximum length of the two waveforms
    max_length = max(waveform1.shape[-1], waveform2.shape[-1])

    # Pad shorter waveform to match the longest one
    if waveform1.shape[-1] < max_length:
        pad_length = max_length - waveform1.shape[-1]
        waveform1 = nnf.pad(waveform1, (0, pad_length))

    if waveform2.shape[-1] < max_length:
        pad_length = max_length - waveform2.shape[-1]
        waveform2 = nnf.pad(waveform2, (0, pad_length))

    # Combine waveforms
    combined_waveform = (waveform1 * weight1) + (waveform2 * weight2)

    # Normalize the combined waveform if necessary
    max_amplitude = torch.max(torch.abs(combined_waveform))
    if max_amplitude > 1.0:
        combined_waveform = combined_waveform / max_amplitude

    return combined_waveform

def calculate_amplitude_envelope(audio, frame_count, frame_rate):
    # Calculate the amplitude envelope of the audio signal
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    # Ensure waveform is a NumPy array for processing
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()

    # Calculate frame length in samples
    frame_length = int(sample_rate / frame_rate)

    amplitude_envelope = []
    for i in range(frame_count):
        start = i * frame_length
        end = start + frame_length
        frame = waveform[start:end]
        if len(frame) == 0:
            amplitude = 0
        else:
            amplitude = np.max(np.abs(frame))
        amplitude_envelope.append(amplitude)

    return amplitude_envelope

def calculate_rms_energy(audio, frame_count, frame_rate):
    # Calculate the RMS energy of the audio signal
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    # Ensure waveform is a NumPy array for processing
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()

    # Calculate frame length in samples
    frame_length = int(sample_rate / frame_rate)

    rms_energy = []
    for i in range(frame_count):
        start = i * frame_length
        end = start + frame_length
        frame = waveform[start:end]
        if len(frame) == 0:
            rms = 0
        else:
            rms = np.sqrt(np.mean(frame ** 2))
        rms_energy.append(rms)

    return rms_energy

def calculate_spectral_flux(audio, frame_count, frame_rate):
    y = audio['waveform']
    sr = audio['sample_rate']
    hop_length = int(sr / frame_rate)
    spectral_flux = []
    prev_spectrum = None
    for i in range(0, len(y), hop_length):
        frame = y[i:i+hop_length]
        spectrum = np.abs(np.fft.fft(frame))
        if prev_spectrum is not None:
            flux = np.sum((spectrum - prev_spectrum) ** 2)
            spectral_flux.append(flux)
        else:
            spectral_flux.append(0)
        prev_spectrum = spectrum
    # Normalize
    spectral_flux = np.array(spectral_flux)
    spectral_flux = spectral_flux / np.max(spectral_flux)
    return spectral_flux[:frame_count]

def calculate_zero_crossing_rate(audio, frame_count, frame_rate):
    """
    Calculate the Zero Crossing Rate (ZCR) of the audio signal.

    Parameters:
    - audio: Dictionary containing 'waveform' and 'sample_rate'.
    - frame_count: Total number of frames to process.
    - frame_rate: Number of frames per second.

    Returns:
    - zero_crossing_rates: List of ZCR values for each frame.
    """
    waveform = audio['waveform']
    sample_rate = audio['sample_rate']

    # Ensure waveform is a NumPy array for processing
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()

    # Flatten the waveform in case it's stereo or multi-channel
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    # Calculate frame length in samples
    frame_length = int(sample_rate / frame_rate)

    zero_crossing_rates = []
    for i in range(frame_count):
        start = i * frame_length
        end = start + frame_length
        frame = waveform[start:end]
        if len(frame) == 0:
            zcr = 0.0
        else:
            # Calculate zero crossings
            zero_crossings = np.where(np.diff(np.sign(frame)))[0]
            zcr = len(zero_crossings) / frame_length
        zero_crossing_rates.append(zcr)

    return zero_crossing_rates