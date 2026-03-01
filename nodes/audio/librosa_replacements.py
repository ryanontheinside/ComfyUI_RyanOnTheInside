"""
Drop-in replacements for librosa functions using only numpy and scipy.
All functions match librosa's API signatures, defaults, and output shapes.
"""

import numpy as np
from scipy.fft import dct
from scipy.ndimage import median_filter
from scipy.signal.windows import hann as scipy_hann


# =============================================================================
# Trivial math conversions
# =============================================================================

def midi_to_hz(notes):
    """Convert MIDI note numbers to frequencies in Hz."""
    notes = np.asarray(notes, dtype=float)
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))


def hz_to_midi(frequencies):
    """Convert frequencies in Hz to MIDI note numbers."""
    frequencies = np.asarray(frequencies, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        midi = 12.0 * (np.log2(frequencies) - np.log2(440.0)) + 69.0
    midi = np.where(np.isfinite(midi), midi, 0.0)
    return midi


def hz_to_note(frequency, cents=False, unicode=True, octave=True):
    """Convert a frequency in Hz to a note name string."""
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if frequency <= 0:
        return 'N/A'
    midi = 12.0 * (np.log2(frequency) - np.log2(440.0)) + 69.0
    midi_rounded = int(np.round(midi))
    note_name = NOTE_NAMES[midi_rounded % 12]
    if octave:
        octave_num = (midi_rounded // 12) - 1
        note_name = f"{note_name}{octave_num}"
    if cents:
        cent_offset = int(np.round(100.0 * (midi - midi_rounded)))
        if cent_offset >= 0:
            note_name = f"{note_name}+{cent_offset:02d}"
        else:
            note_name = f"{note_name}{cent_offset:03d}"
    return note_name


def frames_to_time(frames, sr=22050, hop_length=512):
    """Convert frame indices to time (seconds)."""
    frames = np.asarray(frames, dtype=float)
    return frames * hop_length / sr


def amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0):
    """Convert an amplitude spectrogram to dB-scaled spectrogram."""
    S = np.asarray(S, dtype=float)
    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = np.abs(ref)
    magnitude = np.abs(S)
    power = magnitude ** 2
    ref_power = ref_value ** 2
    log_spec = 10.0 * np.log10(np.maximum(power, amin ** 2))
    log_spec -= 10.0 * np.log10(np.maximum(ref_power, amin ** 2))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


# =============================================================================
# STFT / ISTFT
# =============================================================================

def _get_window(win_length):
    """Get periodic Hann window matching librosa's default."""
    # librosa uses scipy.signal.get_window('hann', win_length, fftbins=True)
    # which is a periodic (DFT-even) Hann window
    return scipy_hann(win_length, sym=False)


def stft(y, n_fft=2048, hop_length=None, win_length=None, center=True):
    """Short-time Fourier Transform matching librosa defaults.
    Supports multi-dimensional input (processes along last axis, matching librosa).
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # Handle multi-dimensional input (matching librosa behavior)
    if np.ndim(y) > 1:
        return np.stack([stft(yi, n_fft=n_fft, hop_length=hop_length,
                              win_length=win_length, center=center)
                         for yi in y])

    window = _get_window(win_length)

    if center:
        y = np.pad(y, n_fft // 2, mode='constant')

    # Pad window to n_fft if needed
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = np.pad(window, (pad_left, pad_right))

    n_frames = 1 + (len(y) - n_fft) // hop_length
    stft_matrix = np.empty((1 + n_fft // 2, n_frames), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + n_fft] * window
        stft_matrix[:, i] = np.fft.rfft(frame, n=n_fft)

    return stft_matrix


def istft(stft_matrix, hop_length=None, win_length=None, center=True, length=None):
    """Inverse Short-time Fourier Transform matching librosa defaults.
    Supports multi-dimensional input (processes along last two axes, matching librosa).
    """
    # Handle multi-dimensional input (matching librosa behavior)
    if stft_matrix.ndim > 2:
        return np.stack([istft(s, hop_length=hop_length, win_length=win_length,
                               center=center, length=length)
                         for s in stft_matrix])

    n_fft = 2 * (stft_matrix.shape[0] - 1)
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    window = _get_window(win_length)

    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        window = np.pad(window, (pad_left, pad_right))

    n_frames = stft_matrix.shape[1]
    expected_length = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_length)
    window_sum = np.zeros(expected_length)

    for i in range(n_frames):
        start = i * hop_length
        frame = np.fft.irfft(stft_matrix[:, i], n=n_fft)
        y[start:start + n_fft] += frame * window
        window_sum[start:start + n_fft] += window ** 2

    nonzero = window_sum > 1e-10
    y[nonzero] /= window_sum[nonzero]

    if center:
        y = y[n_fft // 2:]

    if length is not None:
        y = y[:length]
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))

    return y


# =============================================================================
# Mel filterbank helpers
# =============================================================================

def _hz_to_mel(hz):
    """Convert Hz to mel using Slaney's Auditory Toolbox formula (librosa default)."""
    hz = np.asarray(hz, dtype=float)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp  # 15.0
    logstep = np.log(6.4) / 27.0

    mel = np.where(
        hz < min_log_hz,
        hz / f_sp,
        min_log_mel + np.log(np.maximum(hz, 1e-10) / min_log_hz) / logstep
    )
    return mel


def _mel_to_hz(mel):
    """Convert mel to Hz using Slaney's Auditory Toolbox formula (librosa default)."""
    mel = np.asarray(mel, dtype=float)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    hz = np.where(
        mel < min_log_mel,
        mel * f_sp,
        min_log_hz * np.exp(logstep * (mel - min_log_mel))
    )
    return hz


def _mel_filterbank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None):
    """Create a Mel filterbank matrix matching librosa's mel() with norm='slaney'."""
    if fmax is None:
        fmax = sr / 2.0

    # Compute mel center frequencies (n_mels + 2 points for n_mels filters)
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # Compute ramps: (n_mels+2, n_fft_bins) = mel_hz[i] - fft_freq[j]
    ramps = hz_points[:, np.newaxis] - fft_freqs[np.newaxis, :]
    fdiff = np.diff(hz_points)

    filterbank = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        # Lower slope: (fft_freq - hz[i]) / (hz[i+1] - hz[i])
        lower = -ramps[i] / (fdiff[i] + 1e-10)
        # Upper slope: (hz[i+2] - fft_freq) / (hz[i+2] - hz[i+1])
        upper = ramps[i + 2] / (fdiff[i + 1] + 1e-10)
        filterbank[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney-style normalization
    enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
    filterbank *= enorm[:, np.newaxis]

    return filterbank


def _mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Compute mel spectrogram (power)."""
    S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel_basis = _mel_filterbank(sr, n_fft, n_mels=n_mels)
    return mel_basis @ S


# =============================================================================
# Chroma filterbank
# =============================================================================

def _chroma_filterbank(sr, n_fft, n_chroma=12, tuning=0.0, ctroct=5.0, octwidth=2):
    """Build a chroma filterbank matrix mapping STFT bins to chroma bins.
    Matches librosa.filters.chroma() behavior.
    """
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    n_bins = len(freqs)

    wts = np.zeros((n_chroma, n_bins))

    # Reference frequency: A440 adjusted by tuning
    A440 = 440.0 * 2.0 ** (tuning / n_chroma)

    for i in range(n_bins):
        if freqs[i] <= 0:
            continue
        # Fractional chroma bin (continuous pitch class)
        # A440 = bin 9 (A), so offset by +9 to get C-based chroma (C=0, C#=1, ..., A=9, B=11)
        frac_chroma = (n_chroma * np.log2(freqs[i] / A440) + 9) % n_chroma

        for c in range(n_chroma):
            # Distance in chroma space (circular)
            d = frac_chroma - c
            # Wrap to [-n_chroma/2, n_chroma/2]
            d = d - n_chroma * np.round(d / n_chroma)
            # Gaussian weighting
            wts[c, i] += np.exp(-0.5 * (d / 0.5) ** 2)

        # Octave weighting (downweight very low/high frequencies)
        octs = np.log2(freqs[i] / A440) + 5  # octave number relative to ~13.75 Hz
        oct_weight = np.exp(-0.5 * ((octs - ctroct) / octwidth) ** 2)
        wts[:, i] *= oct_weight

    # Normalize each chroma bin
    norms = np.sqrt(np.sum(wts ** 2, axis=1, keepdims=True))
    wts = wts / (norms + 1e-10)

    return wts


# =============================================================================
# Feature extraction
# =============================================================================

def feature_spectral_centroid(y, sr, n_fft=2048, hop_length=512):
    """Compute spectral centroid. Returns shape (1, n_frames)."""
    S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    centroid = np.sum(freqs[:, np.newaxis] * S, axis=0, keepdims=True) / (np.sum(S, axis=0, keepdims=True) + 1e-10)
    return centroid


def feature_mfcc(y, sr, n_mfcc=20, n_fft=2048, hop_length=512, n_mels=128):
    """Compute MFCCs. Returns shape (n_mfcc, n_frames)."""
    mel_S = _mel_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # librosa uses power_to_db internally for MFCC
    log_mel = 10.0 * np.log10(np.maximum(mel_S, 1e-10))
    mfccs = dct(log_mel, type=2, axis=0, norm='ortho')[:n_mfcc]
    return mfccs


def feature_chroma_stft(y, sr, n_fft=2048, hop_length=512, n_chroma=12):
    """Compute chromagram from STFT. Returns shape (n_chroma, n_frames)."""
    S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    chroma_fb = _chroma_filterbank(sr, n_fft, n_chroma=n_chroma)
    raw_chroma = chroma_fb @ S

    # Normalize each frame (matching librosa's default norm=inf)
    norm = np.max(raw_chroma, axis=0, keepdims=True)
    raw_chroma = raw_chroma / (norm + 1e-10)
    return raw_chroma


def feature_chroma_cqt(y, sr, hop_length=512, n_chroma=12):
    """Approximate chroma_cqt via chroma_stft (adequate for key detection)."""
    return feature_chroma_stft(y, sr, hop_length=hop_length, n_chroma=n_chroma)


def feature_tonnetz(y, sr, n_chroma=12):
    """Compute tonnetz features from chroma. Returns shape (6, n_frames)."""
    chroma = feature_chroma_stft(y, sr, n_chroma=n_chroma)

    # Tonnetz: tonal centroid features (Harte et al.)
    # Use fifths (7 semitones), minor thirds (3), major thirds (4)
    r1 = 1.0   # fifths
    r2 = 1.0   # minor thirds
    r3 = 1.0   # major thirds

    phi = np.arange(n_chroma) * (7.0 * np.pi / 6.0)  # fifths interval
    phi2 = np.arange(n_chroma) * (3.0 * np.pi / 2.0)  # minor third
    phi3 = np.arange(n_chroma) * (2.0 * np.pi / 3.0)  # major third

    tonnetz = np.array([
        r1 * np.sum(chroma * np.sin(phi)[:, np.newaxis], axis=0),
        r1 * np.sum(chroma * np.cos(phi)[:, np.newaxis], axis=0),
        r2 * np.sum(chroma * np.sin(phi2)[:, np.newaxis], axis=0),
        r2 * np.sum(chroma * np.cos(phi2)[:, np.newaxis], axis=0),
        r3 * np.sum(chroma * np.sin(phi3)[:, np.newaxis], axis=0),
        r3 * np.sum(chroma * np.cos(phi3)[:, np.newaxis], axis=0),
    ])

    # Normalize by L1 norm of chroma per frame
    chroma_norm = np.sum(np.abs(chroma), axis=0, keepdims=True) + 1e-10
    tonnetz = tonnetz / chroma_norm
    return tonnetz


def onset_strength(y, sr, hop_length=512, n_fft=2048, n_mels=128, lag=1, center=True):
    """Compute onset strength envelope matching librosa. Returns 1D array."""
    mel_S = _mel_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    # Convert to dB (power_to_db with ref=max, top_db=80)
    S_db = 10.0 * np.log10(np.maximum(mel_S, 1e-10))
    S_db = S_db - np.max(S_db)
    S_db = np.maximum(S_db, -80.0)

    # Spectral flux: S[..., lag:] - ref[..., :-lag], with ref=S (max_size=1)
    onset_env = S_db[:, lag:] - S_db[:, :-lag]

    # Half-wave rectification
    onset_env = np.maximum(0.0, onset_env)

    # Mean across frequency bins
    onset_env = np.mean(onset_env, axis=0)

    # Compensate for lag and centering
    pad_width = lag
    if center:
        pad_width += n_fft // (2 * hop_length)
    onset_env = np.pad(onset_env, (pad_width, 0), mode='constant')

    # Trim to match spectrogram length
    onset_env = onset_env[:S_db.shape[1]]

    return onset_env


# =============================================================================
# Complex algorithms
# =============================================================================

def beat_track(y, sr, hop_length=512, start_bpm=120.0):
    """Estimate tempo and beat positions.
    Returns (tempo, beat_frames) matching librosa's convention.
    """
    onset_env = onset_strength(y, sr, hop_length=hop_length)

    if len(onset_env) < 4:
        return np.array(start_bpm), np.array([], dtype=int)

    # BPM range
    bpm_min, bpm_max = 30.0, 300.0
    sr_onset = sr / hop_length

    lag_min = max(1, int(np.round(60.0 * sr_onset / bpm_max)))
    lag_max = min(len(onset_env) - 1, int(np.round(60.0 * sr_onset / bpm_min)))

    if lag_max <= lag_min:
        return np.array(start_bpm), np.array([], dtype=int)

    # Autocorrelation of onset envelope
    onset_centered = onset_env - np.mean(onset_env)
    corr = np.correlate(onset_centered, onset_centered, mode='full')
    corr = corr[len(onset_centered) - 1:]

    # Prior centered on start_bpm
    start_lag = 60.0 * sr_onset / start_bpm
    lags = np.arange(len(corr), dtype=float)
    bpm_prior = np.exp(-0.5 * ((lags - start_lag) / (start_lag * 0.5)) ** 2)
    weighted_corr = corr * bpm_prior

    search_range = weighted_corr[lag_min:lag_max + 1]
    best_lag = lag_min + np.argmax(search_range)
    tempo = 60.0 * sr_onset / best_lag

    # Beat tracking via peak-picking at expected intervals
    beat_period = best_lag
    beats = []

    threshold = np.mean(onset_env) + 0.5 * np.std(onset_env)
    first_beat = 0
    for i in range(min(beat_period * 2, len(onset_env))):
        if onset_env[i] > threshold:
            first_beat = i
            break

    pos = first_beat
    search_window = max(1, beat_period // 4)
    while pos < len(onset_env):
        start = max(0, pos - search_window)
        end = min(len(onset_env), pos + search_window + 1)
        local_peak = start + np.argmax(onset_env[start:end])
        beats.append(local_peak)
        pos = local_peak + beat_period

    return np.array(tempo), np.array(beats, dtype=int)


def piptrack(y, sr, n_fft=2048, hop_length=None, fmin=150.0, fmax=4000.0, threshold=0.1):
    """Pitch tracking via parabolic interpolation on STFT magnitude peaks.
    Returns (pitches, magnitudes) each of shape (n_fft//2+1, n_frames).
    """
    if hop_length is None:
        hop_length = n_fft // 4

    S = stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(S)
    n_bins, n_frames = mag.shape

    pitches = np.zeros_like(mag)
    magnitudes = np.zeros_like(mag)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    mag_threshold = threshold * np.max(mag)

    for t in range(n_frames):
        for i in range(1, n_bins - 1):
            if freqs[i] < fmin or freqs[i] > fmax:
                continue
            if mag[i, t] < mag_threshold:
                continue
            if mag[i, t] > mag[i - 1, t] and mag[i, t] > mag[i + 1, t]:
                # Parabolic interpolation
                alpha = np.log(mag[i - 1, t] + 1e-10)
                beta = np.log(mag[i, t] + 1e-10)
                gamma = np.log(mag[i + 1, t] + 1e-10)
                denom = alpha - 2 * beta + gamma
                if abs(denom) > 1e-10:
                    p = 0.5 * (alpha - gamma) / denom
                else:
                    p = 0.0
                freq_interp = freqs[i] + p * freq_resolution
                if fmin <= freq_interp <= fmax:
                    pitches[i, t] = freq_interp
                    magnitudes[i, t] = mag[i, t]

    return pitches, magnitudes


def effects_time_stretch(y, rate):
    """Time-stretch audio via phase vocoder."""
    # Handle multi-dimensional input by processing each sub-array
    if y.ndim > 1:
        orig_shape = y.shape[:-1]
        results = []
        for idx in np.ndindex(orig_shape):
            results.append(effects_time_stretch(y[idx], rate))
        # Stack results back into original batch shape
        out_len = results[0].shape[-1]
        out = np.zeros(orig_shape + (out_len,), dtype=y.dtype)
        for idx, r in zip(np.ndindex(orig_shape), results):
            out[idx] = r[:out_len]
        return out

    n_fft = 2048
    hop_length = n_fft // 4

    S = stft(y, n_fft=n_fft, hop_length=hop_length)
    n_bins, n_frames = S.shape

    n_frames_out = int(np.ceil(n_frames / rate))

    # Expected phase advance per hop for each bin
    phase_advance = 2.0 * np.pi * np.arange(n_bins) * hop_length / n_fft

    time_steps = np.arange(n_frames_out) * rate

    S_out = np.zeros((n_bins, n_frames_out), dtype=np.complex128)

    # First frame: use original phase
    phase_acc = np.angle(S[:, 0])
    S_out[:, 0] = np.abs(S[:, 0]) * np.exp(1j * phase_acc)

    for t in range(1, n_frames_out):
        frame_idx = time_steps[t]
        frame_int = int(frame_idx)
        frame_frac = frame_idx - frame_int

        if frame_int + 1 < n_frames:
            mag = (1 - frame_frac) * np.abs(S[:, frame_int]) + frame_frac * np.abs(S[:, frame_int + 1])
            dphi = np.angle(S[:, frame_int + 1]) - np.angle(S[:, frame_int])
        elif frame_int < n_frames:
            mag = np.abs(S[:, frame_int])
            dphi = np.zeros(n_bins)
        else:
            mag = np.zeros(n_bins)
            dphi = np.zeros(n_bins)

        # Unwrap phase difference relative to expected advance
        dphi -= phase_advance
        dphi = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        phase_acc += phase_advance + dphi

        S_out[:, t] = mag * np.exp(1j * phase_acc)

    y_out = istft(S_out, hop_length=hop_length, length=int(len(y) / rate))
    return y_out


# =============================================================================
# HPSS and key-detection chroma
# =============================================================================

def _softmask(X, X_ref, power=2.0):
    """Wiener-style soft mask: X^power / (X^power + X_ref^power)."""
    X_p = np.abs(X) ** power
    X_ref_p = np.abs(X_ref) ** power
    denom = X_p + X_ref_p
    return np.where(denom > 0, X_p / denom, 0.5)


def hpss(S, kernel_size=31):
    """Harmonic-Percussive Source Separation via median filtering.

    Parameters
    ----------
    S : np.ndarray, shape (n_freq, n_frames)
        Magnitude spectrogram (non-negative).
    kernel_size : int
        Size of the median filter kernel (must be odd).

    Returns
    -------
    H : np.ndarray  -- harmonic component spectrogram
    P : np.ndarray  -- percussive component spectrogram
    """
    k = kernel_size
    # Harmonic: median along time axis (horizontal)
    H_med = median_filter(S, size=(1, k))
    # Percussive: median along frequency axis (vertical)
    P_med = median_filter(S, size=(k, 1))
    # Soft-mask the original spectrogram
    mask_H = _softmask(H_med, P_med)
    mask_P = _softmask(P_med, H_med)
    return S * mask_H, S * mask_P


def estimate_tuning(y, sr, n_fft=2048, fmin=150.0, fmax=4000.0):
    """Estimate tuning offset in fractions of a semitone.

    Returns a float in [-0.5, 0.5). Falls back to 0.0 if no pitched
    content is detected.
    """
    pitches, mags = piptrack(y, sr, n_fft=n_fft, fmin=fmin, fmax=fmax)
    # Keep only nonzero pitch estimates
    mask = pitches > 0
    if not np.any(mask):
        return 0.0
    p = pitches[mask]
    # Fractional chroma residual relative to A440 (semitones mod 1)
    residuals = (12.0 * np.log2(p / 440.0)) % 1.0
    # Map to [-0.5, 0.5)
    residuals = np.where(residuals >= 0.5, residuals - 1.0, residuals)
    # Histogram to find dominant offset
    bins = np.linspace(-0.5, 0.5, 65)  # 64 bins
    counts, edges = np.histogram(residuals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[np.argmax(counts)])


def cqt(y, sr, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0):
    """Constant-Q Transform via time-domain spectral kernels.

    Builds a kernel matrix of windowed complex exponentials (one per bin)
    and computes the transform via batched matrix multiplication.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (1D).
    sr : int
        Sample rate.
    hop_length : int
        Hop length in samples.
    fmin : float or None
        Minimum frequency. Defaults to C1 (~32.7 Hz).
    n_bins : int
        Number of frequency bins.
    bins_per_octave : int
        Number of bins per octave.
    tuning : float
        Tuning offset in fractions of a bin.

    Returns
    -------
    C : np.ndarray, shape (n_bins, n_frames)
        Complex CQT matrix. Bin 0 = fmin, bin bins_per_octave = fmin*2, etc.
    """
    if fmin is None:
        fmin = midi_to_hz(24)  # C1 ~32.7 Hz
    # Apply tuning offset
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # Q factor for constant-Q spacing
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)

    # Center frequencies and window lengths for each bin
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
    lengths = np.ceil(Q * sr / freqs).astype(int)
    max_len = int(lengths[0])  # lowest frequency bin has longest window

    # Build kernel matrix (n_bins, max_len)
    # Each row is a centered, windowed complex exponential normalized by 1/N_k
    kernel = np.zeros((n_bins, max_len), dtype=np.complex128)
    for k in range(n_bins):
        N_k = int(lengths[k])
        n = np.arange(N_k)
        window = scipy_hann(N_k, sym=False)
        start = (max_len - N_k) // 2
        kernel[k, start:start + N_k] = (
            window * np.exp(-2j * np.pi * freqs[k] * n / sr) / N_k
        )

    # Center-pad the signal
    y_padded = np.pad(y, max_len // 2, mode='constant')
    n_frames = 1 + (len(y_padded) - max_len) // hop_length

    # Batched matrix multiplication in chunks to limit memory
    cqt_out = np.empty((n_bins, n_frames), dtype=np.complex128)
    chunk_size = 2048
    win_idx = np.arange(max_len)
    for c_start in range(0, n_frames, chunk_size):
        c_end = min(c_start + chunk_size, n_frames)
        offsets = np.arange(c_start, c_end) * hop_length
        frames = y_padded[offsets[:, None] + win_idx[None, :]]  # (n_chunk, max_len)
        cqt_out[:, c_start:c_end] = kernel @ frames.T

    return cqt_out


def feature_chroma_for_key_detection(y, sr, n_fft=4096, hop_length=512):
    """Compute an energy-weighted 12-element chroma profile optimized for
    key detection.

    Uses CQT (fmin=C2, 6 octaves) for proper pitch resolution at all
    frequencies, HPSS (harmonic only), and tuning compensation.

    Returns
    -------
    chroma_profile : np.ndarray, shape (12,)
        Energy-weighted chroma vector (not per-frame).
    """
    # Estimate tuning from the raw audio
    tuning = estimate_tuning(y, sr)

    # CQT: fmin=C2 (MIDI 36, ~65.4 Hz), 72 bins = 6 octaves (C2-B7)
    fmin_c2 = midi_to_hz(36)
    C = cqt(y, sr, hop_length=hop_length, fmin=fmin_c2, n_bins=72,
            bins_per_octave=12, tuning=tuning)

    # HPSS on CQT magnitude -- keep harmonic component only
    C_mag = np.abs(C)
    C_harm, _ = hpss(C_mag)

    # Power of harmonic component
    C_harm_power = C_harm ** 2

    # Fold octaves into 12 chroma bins by summing every 12th row
    # Bin 0 = C2, bin 1 = C#2, ..., bin 12 = C3, etc.
    # Chroma index = bin_index % 12, with index 0 = C
    chroma = np.zeros((12, C_harm_power.shape[1]))
    for i in range(12):
        chroma[i] = np.sum(C_harm_power[i::12], axis=0)

    # Energy per frame (for weighting)
    frame_energy = np.sum(C_harm_power, axis=0)

    # Weighted average across time
    total_energy = np.sum(frame_energy)
    if total_energy < 1e-10:
        return np.ones(12) / 12.0

    chroma_profile = chroma @ frame_energy / total_energy
    return chroma_profile


# =============================================================================
# Display wrappers (matplotlib)
# =============================================================================

def display_specshow(data, sr=22050, x_axis=None, y_axis=None, cmap=None, ax=None):
    """Display a spectrogram-like array using matplotlib imshow."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    kwargs = {}
    if cmap is not None:
        kwargs['cmap'] = cmap

    img = ax.imshow(data, aspect='auto', origin='lower', interpolation='nearest', **kwargs)

    n_y, n_x = data.shape

    if x_axis == 'time':
        hop_length = 512
        times = np.linspace(0, n_x * hop_length / sr, num=5)
        ax.set_xticks(np.linspace(0, n_x, num=5))
        ax.set_xticklabels([f'{t:.2f}' for t in times])
        ax.set_xlabel('Time (s)')
    elif x_axis == 'off':
        pass

    if y_axis == 'hz' or y_axis == 'linear':
        freqs = np.linspace(0, sr / 2, num=5)
        ax.set_yticks(np.linspace(0, n_y, num=5))
        ax.set_yticklabels([f'{f:.0f}' for f in freqs])
        ax.set_ylabel('Hz')
    elif y_axis == 'log':
        ax.set_ylabel('Hz (log)')
    elif y_axis == 'mel':
        ax.set_ylabel('Mel')
    elif y_axis == 'off':
        pass

    return img


def display_waveshow(y, sr=22050, ax=None, x_axis=None):
    """Display a waveform using matplotlib plot."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()

    times = np.arange(len(y)) / sr
    ax.plot(times, y)

    if x_axis == 'time' or x_axis is None:
        ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    return ax
