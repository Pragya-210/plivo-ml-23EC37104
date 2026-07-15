#!/usr/bin/env python3
"""
Causal audio feature extraction for End-of-Turn Detection.
All features use ONLY audio from time 0 up to pause_start.
Allowed: numpy, scipy, pandas, scikit-learn
"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.fft import rfft, rfftfreq


def load_audio(path, target_sr=16000):
    """Load and normalize audio to target sample rate."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        num_samples = int(len(data) * target_sr / sr)
        data = signal.resample(data.astype(np.float64), num_samples)
        sr = target_sr
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
        mx = np.max(np.abs(data))
        if mx > 1.0:
            data = data / mx
    return sr, data


def detect_pauses(audio, sr, min_pause_ms=100, energy_threshold_db=50):
    """
    Detect silence pauses using energy-based VAD.
    Tuned to match ground truth annotation style.
    """
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    
    n_frames = (len(audio) - frame_length) // hop_length + 1
    energies = np.zeros(n_frames)
    frame_starts = np.arange(n_frames) * hop_length
    
    for i in range(n_frames):
        s = i * hop_length
        e = s + frame_length
        if e <= len(audio):
            energies[i] = np.mean(audio[s:e]**2)
    
    log_energies = 10 * np.log10(energies + 1e-10)
    max_energy = np.max(log_energies)
    threshold = max_energy - energy_threshold_db
    
    is_speech = log_energies > threshold
    
    # Skip leading silence
    first_speech = np.where(is_speech)[0]
    if len(first_speech) > 0:
        start_frame = first_speech[0]
        is_speech = is_speech[start_frame:]
        frame_starts = frame_starts[start_frame:]
    
    min_frames = int(min_pause_ms / 10)
    pauses = []
    in_pause = False
    pause_start_idx = None
    
    for i, speech in enumerate(is_speech):
        if not speech and not in_pause:
            in_pause = True
            pause_start_idx = i
        elif speech and in_pause:
            in_pause = False
            pause_end_idx = i
            if pause_end_idx - pause_start_idx >= min_frames:
                start_time = frame_starts[pause_start_idx] / sr
                end_time = frame_starts[pause_end_idx] / sr
                pauses.append({
                    'pause_start': start_time,
                    'pause_end': end_time,
                    'duration_ms': (end_time - start_time) * 1000
                })
    
    if in_pause:
        pause_end_idx = len(is_speech)
        if pause_end_idx - pause_start_idx >= min_frames:
            start_time = frame_starts[pause_start_idx] / sr
            end_time = len(audio) / sr
            pauses.append({
                'pause_start': start_time,
                'pause_end': end_time,
                'duration_ms': (end_time - start_time) * 1000
            })
    
    return pauses


def extract_pitch_fast(audio, sr, fmin=50, fmax=500, frame_length=2048, hop_length=512):
    """Fast autocorrelation-based pitch extraction."""
    n_frames = max(0, (len(audio) - frame_length) // hop_length)
    if n_frames == 0:
        return np.array([]), np.array([])
    
    pitches = np.zeros(n_frames)
    confidences = np.zeros(n_frames)
    lag_min = max(1, int(sr / fmax))
    lag_max = min(int(sr / fmin), frame_length - 1)
    
    for i in range(n_frames):
        frame = audio[i*hop_length : i*hop_length + frame_length]
        frame = frame * np.hanning(len(frame))
        
        fft_len = 2 * frame_length
        spec = rfft(frame, n=fft_len)
        autocorr = np.fft.irfft(spec * np.conj(spec), n=fft_len)[:frame_length]
        
        search_region = autocorr[lag_min:lag_max+1]
        if len(search_region) > 0 and np.max(search_region) > 0:
            peak_idx = lag_min + np.argmax(search_region)
            if peak_idx > 0 and peak_idx < len(autocorr) - 1:
                a = autocorr[peak_idx - 1]
                b = autocorr[peak_idx]
                c = autocorr[peak_idx + 1]
                denom = a + c - 2*b
                if denom != 0:
                    p = 0.5 * (a - c) / denom
                    peak_idx = peak_idx + p
            
            pitch = sr / peak_idx if peak_idx > 0 else 0
            conf = autocorr[int(peak_idx)] / (autocorr[0] + 1e-10)
            
            if fmin <= pitch <= fmax and conf > 0.3:
                pitches[i] = pitch
                confidences[i] = conf
    
    return pitches, confidences


def extract_features(audio, sr, pause_start, window_before=1.5):
    """Extract causal features from window_before seconds before pause."""
    end_sample = int(pause_start * sr)
    start_sample = max(0, end_sample - int(window_before * sr))
    
    if end_sample - start_sample < int(0.1 * sr):
        return None
    
    speech = audio[start_sample:end_sample]
    duration = len(speech) / sr
    features = {}
    
    # Energy features
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    n_frames = max(1, (len(speech) - frame_len) // hop_len + 1)
    
    energies = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * hop_len
        e = s + frame_len
        if e <= len(speech):
            energies[i] = np.mean(speech[s:e]**2)
    
    log_energies = 10 * np.log10(energies + 1e-10)
    features['energy_mean'] = float(np.mean(log_energies))
    features['energy_std'] = float(np.std(log_energies))
    features['energy_last'] = float(log_energies[-1])
    features['energy_trend'] = float(np.polyfit(np.arange(len(log_energies)), log_energies, 1)[0]) if len(log_energies) > 1 else 0.0
    features['energy_drop'] = float(log_energies[0] - log_energies[-1]) if len(log_energies) > 1 else 0.0
    features['energy_ratio_last_mean'] = float(energies[-1] / (np.mean(energies) + 1e-10))
    
    # Pitch features
    pitches, confidences = extract_pitch_fast(speech, sr)
    valid = confidences > 0.5
    valid_pitches = pitches[valid]
    
    if len(valid_pitches) > 0:
        features['pitch_mean'] = float(np.mean(valid_pitches))
        features['pitch_std'] = float(np.std(valid_pitches))
        features['pitch_last'] = float(valid_pitches[-1])
        features['pitch_trend'] = float(np.polyfit(np.arange(len(valid_pitches)), valid_pitches, 1)[0]) if len(valid_pitches) > 1 else 0.0
        features['pitch_drop'] = float(valid_pitches[0] - valid_pitches[-1]) if len(valid_pitches) > 1 else 0.0
        features['pitch_range'] = float(np.max(valid_pitches) - np.min(valid_pitches))
        features['voiced_ratio'] = float(len(valid_pitches) / max(len(pitches), 1))
    else:
        features['pitch_mean'] = 0.0
        features['pitch_std'] = 0.0
        features['pitch_last'] = 0.0
        features['pitch_trend'] = 0.0
        features['pitch_drop'] = 0.0
        features['pitch_range'] = 0.0
        features['voiced_ratio'] = 0.0
    
    # Spectral features
    features['zcr'] = float(np.mean(np.abs(np.diff(np.sign(speech)))) / 2.0)
    
    if len(speech) > 256:
        spec = np.abs(rfft(speech))
        freqs = rfftfreq(len(speech), 1/sr)
        features['spectral_centroid'] = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-10))
        cumsum = np.cumsum(spec)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        features['spectral_rolloff'] = float(freqs[min(rolloff_idx, len(freqs)-1)])
    else:
        features['spectral_centroid'] = 0.0
        features['spectral_rolloff'] = 0.0
    
    # Duration feature
    features['speech_before_duration'] = float(duration)
    
    return features
