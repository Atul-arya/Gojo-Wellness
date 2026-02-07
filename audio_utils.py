
import numpy as np
import wave
import io
from typing import Tuple, Optional, List

def load_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Load raw audio bytes (WAV) into numpy array and sample rate.
    
    Args:
        audio_bytes: Binary content of WAV file
        
    Returns:
        (audio_data, sample_rate)
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            raw_data = wf.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            audio = np.frombuffer(raw_data, dtype=dtype)
            
            # Convert to float [-1, 1]
            if sample_width == 1:
                audio = (audio.astype(np.float32) - 128) / 128.0
            else:
                audio = audio.astype(np.float32) / (2**(8*sample_width - 1))
            
            # Mix to mono if stereo
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
                
            return audio, framerate
            
    except Exception as e:
        print(f"Error loading audio: {e}")
        return np.array([]), 0

def extract_pitch(audio: np.ndarray, sr: int, frame_size_ms: int = 30, hop_size_ms: int = 15) -> np.ndarray:
    """
    Extract pitch (F0) using autocorrelation method.
    Optimized for voice range (50Hz - 500Hz).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        frame_size_ms: Window size (ms)
        hop_size_ms: Step size (ms)
        
    Returns:
        Array of F0 values (0 where unvoiced)
    """
    if len(audio) == 0:
        return np.array([])

    frame_length = int(sr * frame_size_ms / 1000)
    hop_length = int(sr * hop_size_ms / 1000)
    
    num_frames = max(1, (len(audio) - frame_length) // hop_length)
    pitches = []
    
    # Human voice range constraints
    f_min = 50   # Hz
    f_max = 500  # Hz
    
    lag_min = int(sr / f_max)
    lag_max = int(sr / f_min)
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        
        # Simple energy check for voicing
        if np.sum(frame**2) < 0.001:  # Silence
            pitches.append(0.0)
            continue
            
        # Autocorrelation
        # Use FFT for speed: correlate(x, x) = ifft(fft(x) * conj(fft(x)))
        try:
            # Zero-pad for linear convolution
            n = len(frame)
            padded = np.pad(frame, (0, n), 'constant')
            
            f = np.fft.fft(padded)
            corr = np.fft.ifft(f * np.conjugate(f)).real
            corr = corr[:n]  # Keep positive lags
            
            # Normalize
            if corr[0] == 0:
                pitches.append(0.0)
                continue
            
            # Find peak in valid range
            # We look for the first major peak after lag_min
            valid_corr = corr[lag_min:lag_max]
            if len(valid_corr) == 0:
                pitches.append(0.0)
                continue
                
            peak_idx = np.argmax(valid_corr)
            peak_val = valid_corr[peak_idx]
            
            # Refine lag
            lag = lag_min + peak_idx
            
            # Quality check: peak must be significant relative to energy (corr[0])
            if peak_val / corr[0] > 0.3:  # Valid periodicity
                f0 = sr / lag
                pitches.append(f0)
            else:
                pitches.append(0.0)
                
        except Exception:
            pitches.append(0.0)
            
    return np.array(pitches)
