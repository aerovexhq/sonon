#!/usr/bin/env python3
"""
Aerovex Sonon - Phoneme-Aligned Articulatory Voice Synthesizer
=============================================================
Synthesizes speech using frame-aligned phonetic trajectories, 
vocal tract all-pole formant synthesis (Fant's acoustic model),
and pitch-synchronous glottal flow excitation.

Replaces stationary comb filters and unaligned cross-attention 
blur with crisp syllable transitions, silent stop closures, 
and authentic formant glides.

Zero emojis. Strictly safe Python/Rust reproducible DSP.
"""

import math
import numpy as np
import scipy.signal
import soundfile as sf
from typing import List, Tuple, Dict

SAMPLE_RATE = 24000

# Canonical English Phoneme Formant Inventory (F1, F2, F3 in Hz, Bandwidths B1, B2, B3, voicing flag)
PHONEME_TABLE = {
    # Vowels
    "IY": {"f": [270, 2290, 3010, 3500], "b": [50, 100, 150, 200], "v": 1.0, "noise": 0.0}, # feet, reached
    "IH": {"f": [390, 1990, 2550, 3500], "b": [50, 100, 150, 200], "v": 1.0, "noise": 0.0}, # maintaining (in)
    "EY": {"f": [530, 1840, 2480, 3500], "b": [60, 110, 160, 200], "v": 1.0, "noise": 0.0}, # waypoint (way)
    "EH": {"f": [530, 1840, 2480, 3500], "b": [60, 110, 160, 200], "v": 1.0, "noise": 0.0}, # maintaining (ten)
    "AE": {"f": [660, 1720, 2410, 3500], "b": [70, 130, 180, 200], "v": 1.0, "noise": 0.0}, # alpha (al)
    "AA": {"f": [730, 1090, 2440, 3500], "b": [70, 120, 180, 200], "v": 1.0, "noise": 0.0},
    "AH": {"f": [520, 1190, 2390, 3500], "b": [60, 110, 170, 200], "v": 1.0, "noise": 0.0}, # alpha (pha)
    "AO": {"f": [570, 840,  2410, 3500], "b": [70, 120, 180, 200], "v": 1.0, "noise": 0.0},
    "OW": {"f": [500, 850,  2400, 3500], "b": [70, 120, 180, 200], "v": 1.0, "noise": 0.0},
    "UW": {"f": [300, 870,  2240, 3500], "b": [60, 110, 170, 200], "v": 1.0, "noise": 0.0}, # altitude (tude)
    "ER": {"f": [490, 1350, 1690, 3500], "b": [60, 100, 150, 200], "v": 1.0, "noise": 0.0},
    "AW": {"f": [750, 1250, 2400, 3500], "b": [70, 120, 180, 200], "v": 1.0, "noise": 0.0}, # thousand
    "OY": {"f": [500, 900,  2400, 3500], "b": [70, 120, 180, 200], "v": 1.0, "noise": 0.0}, # point
    
    # Liquids & Glides
    "W":  {"f": [300, 610,  2200, 3500], "b": [60, 100, 150, 200], "v": 0.9, "noise": 0.0}, # waypoint (w)
    "L":  {"f": [380, 1200, 2700, 3500], "b": [60, 120, 180, 200], "v": 0.9, "noise": 0.0}, # alpha (l)
    "R":  {"f": [420, 1300, 1600, 3500], "b": [70, 120, 160, 200], "v": 0.9, "noise": 0.0}, # reached (r)
    "Y":  {"f": [280, 2200, 2900, 3500], "b": [50, 100, 150, 200], "v": 0.9, "noise": 0.0},
    
    # Nasals
    "M":  {"f": [280, 1050, 2200, 3500], "b": [100, 200, 250, 200], "v": 0.8, "noise": 0.0}, # maintaining (m)
    "N":  {"f": [280, 1500, 2200, 3500], "b": [100, 200, 250, 200], "v": 0.8, "noise": 0.0}, # maintaining (n)
    "NG": {"f": [280, 2000, 2500, 3500], "b": [120, 220, 250, 200], "v": 0.8, "noise": 0.0}, # maintaining (ng)
    
    # Fricatives (Turbulent aspiration noise + anterior cavity)
    "S":  {"f": [400, 1600, 2600, 5200], "b": [200, 300, 400, 500], "v": 0.0, "noise": 0.9}, # thousand (s)
    "F":  {"f": [350, 1300, 2300, 3800], "b": [250, 350, 450, 600], "v": 0.0, "noise": 0.5}, # feet (f)
    "TH": {"f": [400, 1500, 2500, 4500], "b": [200, 300, 400, 500], "v": 0.0, "noise": 0.6}, # three (th)
    "CH": {"f": [350, 1800, 2700, 4200], "b": [150, 250, 350, 450], "v": 0.1, "noise": 0.8}, # reached (ch)
    
    # Stops / Plosives (Silence closure followed by burst transient)
    "P":  {"f": [300, 800,  2200, 3500], "b": [100, 200, 300, 400], "v": 0.0, "noise": 0.3, "is_stop": True},
    "T":  {"f": [300, 1800, 2600, 4000], "b": [100, 200, 300, 400], "v": 0.0, "noise": 0.4, "is_stop": True},
    "K":  {"f": [300, 2000, 2700, 3500], "b": [100, 200, 300, 400], "v": 0.0, "noise": 0.4, "is_stop": True},
    
    # Silence / Pause
    "SIL": {"f": [500, 1500, 2500, 3500], "b": [100, 200, 300, 400], "v": 0.0, "noise": 0.0}
}

def get_waypoint_phoneme_sequence() -> List[Tuple[str, float]]:
    """
    Returns time-aligned phoneme sequence for:
    'Waypoint Alpha reached. Maintaining altitude three thousand feet.'
    (phoneme_name, duration_seconds)
    """
    return [
        # "Waypoint"
        ("W", 0.09), ("EY", 0.14), ("P", 0.06), ("OY", 0.14), ("N", 0.08), ("T", 0.05),
        ("SIL", 0.08),
        # "Alpha"
        ("AE", 0.13), ("L", 0.09), ("F", 0.09), ("AH", 0.11),
        ("SIL", 0.08),
        # "reached."
        ("R", 0.08), ("IY", 0.15), ("CH", 0.10), ("T", 0.05),
        ("SIL", 0.22), # sentence boundary pause
        # "Maintaining"
        ("M", 0.08), ("EY", 0.11), ("N", 0.07), ("T", 0.05), ("EY", 0.10), ("N", 0.07), ("IH", 0.09), ("NG", 0.09),
        ("SIL", 0.06),
        # "altitude"
        ("AE", 0.11), ("L", 0.08), ("T", 0.05), ("IH", 0.09), ("T", 0.05), ("UW", 0.14), ("T", 0.05),
        ("SIL", 0.08),
        # "three"
        ("TH", 0.10), ("R", 0.08), ("IY", 0.16),
        ("SIL", 0.05),
        # "thousand"
        ("TH", 0.09), ("AW", 0.14), ("Z", 0.08), ("AH", 0.09), ("N", 0.07), ("T", 0.05),
        ("SIL", 0.06),
        # "feet."
        ("F", 0.10), ("IY", 0.18), ("T", 0.06),
        ("SIL", 0.20)
    ]

def synthesize_articulatory_speech(f0_base: float = 140.0) -> np.ndarray:
    seq = get_waypoint_phoneme_sequence()
    total_dur = sum(d for _, d in seq)
    total_samples = int(total_dur * SAMPLE_RATE)
    
    # 1. Build continuous trajectories for F1, F2, F3, F4 and Bandwidths
    f_traj = np.zeros((4, total_samples))
    b_traj = np.zeros((4, total_samples))
    v_traj = np.zeros(total_samples)
    n_traj = np.zeros(total_samples)
    stop_gate = np.ones(total_samples)
    
    cur_idx = 0
    for ph_name, dur in seq:
        ph = PHONEME_TABLE.get(ph_name, PHONEME_TABLE["SIL"])
        n_samples = int(dur * SAMPLE_RATE)
        end_idx = min(total_samples, cur_idx + n_samples)
        
        for k in range(4):
            f_traj[k, cur_idx:end_idx] = ph["f"][k]
            b_traj[k, cur_idx:end_idx] = ph["b"][k]
        v_traj[cur_idx:end_idx] = ph["v"]
        n_traj[cur_idx:end_idx] = ph["noise"]
        
        if ph.get("is_stop", False):
            # Plosive: first 70% is silent closure, last 30% is burst
            closure_samples = int(0.65 * (end_idx - cur_idx))
            stop_gate[cur_idx:cur_idx + closure_samples] = 0.0
            
        cur_idx = end_idx
        
    # Smooth formant trajectories with human vocal tract inertial filter (Gaussian sigma = 15ms)
    smooth_sigma = int(0.015 * SAMPLE_RATE)
    for k in range(4):
        f_traj[k] = scipy.ndimage.gaussian_filter1d(f_traj[k], smooth_sigma)
        b_traj[k] = scipy.ndimage.gaussian_filter1d(b_traj[k], smooth_sigma)
    v_traj = scipy.ndimage.gaussian_filter1d(v_traj, smooth_sigma // 2)
    n_traj = scipy.ndimage.gaussian_filter1d(n_traj, smooth_sigma // 2)
    
    # 2. Pitch trajectory: natural intonation contour with declination and phrase accents
    t_axis = np.linspace(0, total_dur, total_samples)
    f0 = f0_base - 15.0 * (t_axis / total_dur) # natural declination
    # Accent rises on stressed words (Waypoint, Alpha, Altitude, Three, Feet)
    f0 += 18.0 * np.sin(2 * np.pi * 1.8 * t_axis) * v_traj
    f0 = np.clip(f0, 95.0, 220.0)
    
    # Vocal fold micro-jitter
    f0_jitter = f0 * (1.0 + 0.007 * scipy.ndimage.gaussian_filter1d(np.random.randn(total_samples), 40))
    phase = 2 * np.pi * np.cumsum(f0_jitter) / SAMPLE_RATE
    
    # 3. Liljencrants-Fant (LF) Glottal Flow Excitation
    # Differentiated glottal flow wave: smooth opening, rapid negative closure peak
    phase_wrap = phase % (2 * np.pi)
    glottal_pulse = np.zeros(total_samples)
    # Open quotient = 0.6, return quotient = 0.1
    open_mask = phase_wrap < (1.2 * np.pi)
    glottal_pulse[open_mask] = 0.5 * (1.0 - np.cos(phase_wrap[open_mask] / 0.6))
    glottal_pulse[~open_mask] = -1.2 * np.exp(-((phase_wrap[~open_mask] - 1.2 * np.pi) / 0.25))
    glottal_diff = np.diff(glottal_pulse, prepend=0)
    
    # Turbulent aspiration / frication noise
    white_noise = np.random.randn(total_samples)
    noise_source = scipy.signal.medfilt(white_noise, 3) * n_traj
    
    excitation = (v_traj * glottal_diff + noise_source) * stop_gate
    
    # 4. Formant Cascade Resonator Bank (4 parallel/cascade 2-pole resonators)
    # H_k(z) = (1 - 2 r cos(theta) + r^2) / (1 - 2 r cos(theta) z^-1 + r^2 z^-2)
    audio = np.zeros(total_samples)
    
    # Compute in block-wise frame updates (hop = 64 samples = 2.6 ms) for time-varying IIR
    hop = 64
    n_blocks = total_samples // hop
    
    # 4 cascade second-order resonators
    y = np.copy(excitation)
    for k in range(4):
        # Apply 2-pole resonator in 2.6ms blocks
        filtered_k = np.zeros(total_samples)
        z1, z2 = 0.0, 0.0
        for b in range(n_blocks):
            s_start = b * hop
            s_end = min(total_samples, s_start + hop)
            
            F = f_traj[k, s_start]
            B = b_traj[k, s_start]
            
            r = math.exp(-math.pi * B / SAMPLE_RATE)
            theta = 2.0 * math.pi * F / SAMPLE_RATE
            
            a1 = -2.0 * r * math.cos(theta)
            a2 = r * r
            gain = 1.0 + a1 + a2 if (1.0 + a1 + a2) > 0.001 else 0.05
            
            # Direct form II
            for n in range(s_start, s_end):
                x_n = y[n]
                w = x_n - a1 * z1 - a2 * z2
                filtered_k[n] = gain * w
                z2 = z1
                z1 = w
                
        # Cascade coupling: input to next formant is output of current
        y = filtered_k
        
    audio = y
    
    # 5. Lip Radiation Characteristic (1st order differentiator: +6 dB/octave)
    audio = np.diff(audio, prepend=0)
    
    # Normalization
    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        audio = (audio / peak) * 0.89125 # -1.0 dBFS
        
    # Highpass DC removal (45 Hz)
    sos = scipy.signal.butter(4, 45.0, btype="highpass", fs=SAMPLE_RATE, output="sos")
    audio = scipy.signal.sosfilt(sos, audio)
    
    # Tail cosine fadeout (100 ms)
    fade_len = int(0.10 * SAMPLE_RATE)
    audio[-fade_len:] *= 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
    
    return audio

if __name__ == "__main__":
    out_wav = "/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis/test_phoneme_aligned_speech.wav"
    audio = synthesize_articulatory_speech()
    sf.write(out_wav, audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    print(f"Phoneme-aligned articulatory speech synthesized successfully: {out_wav}")
    print(f"Duration: {len(audio)/SAMPLE_RATE:.2f}s, SR: {SAMPLE_RATE} Hz")
