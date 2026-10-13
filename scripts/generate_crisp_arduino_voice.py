import numpy as np
import scipy.signal as signal
import soundfile as sf

ref_path = '/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis/sonon_neural_waypoint_reached.wav'
ref_audio, sr = sf.read(ref_path)

order = 18
frame_len = int(0.020 * sr) # 20ms
hop_len = int(0.005 * sr)   # 5ms
n_frames = (len(ref_audio) - frame_len) // hop_len

# Gentle 40 Hz subsonic filter
sos_hp = signal.butter(2, 40, btype='highpass', fs=sr, output='sos')
clean_ref = signal.sosfilt(sos_hp, ref_audio)

lpc_coeffs = []
f0_list = []
gains = []

for i in range(n_frames):
    start = i * hop_len
    frame = clean_ref[start:start+frame_len] * np.hanning(frame_len)
    r = np.correlate(frame, frame, mode='full')[frame_len-1:]
    
    if r[0] > 1e-6:
        a_prev = np.array([1.0])
        e = r[0]
        for p in range(1, order + 1):
            gamma = r[p]
            for j in range(1, p):
                gamma += a_prev[j] * r[p - j]
            k_val = np.clip(-gamma / e, -0.985, 0.985)
            a_cur = np.zeros(p + 1)
            a_cur[0] = 1.0
            for j in range(1, p):
                a_cur[j] = a_prev[j] + k_val * a_prev[p - j]
            a_cur[p] = k_val
            e *= (1.0 - k_val**2)
            a_prev = a_cur
        
        # Bandwidth sharpening: gamma = 0.985 for crisp vocal tract definition
        gamma_factors = np.array([0.985**k for k in range(order + 1)])
        lpc_coeffs.append(a_cur * gamma_factors)
        
        # Gain is residual standard deviation: sqrt(e / frame_len)
        gain = np.sqrt(e / frame_len)
        gains.append(gain)
    else:
        lpc_coeffs.append(np.array([1.0] + [0.0]*order))
        gains.append(0.0)
        
    min_lag = int(sr / 340.0)
    max_lag = int(sr / 80.0)
    if r[0] > 1e-4 and max_lag < len(r):
        peak_lag = min_lag + np.argmax(r[min_lag:max_lag])
        peak_ratio = r[peak_lag] / r[0]
        f0 = (sr / peak_lag) if peak_ratio > 0.28 else 0.0
    else:
        f0 = 0.0
    f0_list.append(f0)

# Resynthesis with sample-by-sample continuous interpolation and smooth onset phase reset
out_audio = np.zeros(n_frames * hop_len)
filter_state = np.zeros(order)
phase = 0.0
was_voiced = False

prev_gain = gains[0]
prev_a = lpc_coeffs[0]

for i in range(n_frames):
    start = i * hop_len
    end = start + hop_len
    f0 = f0_list[i]
    target_gain = gains[i]
    target_a = lpc_coeffs[i]
    
    is_voiced = (f0 > 45.0)
    # Smooth phase initialization at voiced onsets (eliminates onset click)
    if is_voiced and not was_voiced:
        phase = 0.0
    was_voiced = is_voiced
    
    exc = np.zeros(hop_len)
    if is_voiced:
        phase_inc = f0 / sr
        for s in range(hop_len):
            phase += phase_inc
            if phase >= 1.0:
                phase -= 1.0
            # Natural Liljencrants-Fant glottal flow derivative (open + return phase)
            if phase < 0.60:
                pulse = np.sin(np.pi * phase / 0.60)**2
            elif phase < 0.80:
                t_rel = (phase - 0.60) / 0.20
                pulse = -1.0 * np.exp(-3.5 * t_rel) * np.sin(np.pi * t_rel)
            else:
                pulse = 0.0
            # Glottal breath aspiration for air presence
            noise = (np.random.rand() * 2.0 - 1.0) * 0.08
            exc[s] = pulse + noise
    else:
        # Natural unvoiced frication
        exc = (np.random.rand(hop_len) * 2.0 - 1.0)
        
    rms_exc = np.sqrt(np.mean(exc**2)) + 1e-6
    exc = exc / rms_exc
    
    frame_out = np.zeros(hop_len)
    for s in range(hop_len):
        # Cosine interpolation (smooth S-curve, continuous 1st derivative)
        alpha = 0.5 * (1.0 - np.cos(np.pi * s / float(hop_len)))
        cur_gain = (1.0 - alpha) * prev_gain + alpha * target_gain
        cur_a = (1.0 - alpha) * prev_a + alpha * target_a
        
        in_sample = exc[s] * cur_gain
        y_val = in_sample
        for p in range(order):
            y_val -= cur_a[p + 1] * filter_state[p]
        for p in range(order - 1, 0, -1):
            filter_state[p] = filter_state[p - 1]
        filter_state[0] = y_val
        frame_out[s] = y_val
        
    prev_gain = target_gain
    prev_a = target_a
    out_audio[start:end] = frame_out

out_audio = signal.sosfilt(sos_hp, out_audio)

# Peak normalize to -1.0 dBFS
peak = np.max(np.abs(out_audio))
if peak > 1e-4:
    out_audio = out_audio * (0.891 / peak)

diff = np.abs(np.diff(out_audio))
spikes_08 = np.where(diff > 0.8)[0]
spikes_045 = np.where(diff > 0.45)[0]

b_air, a_air = signal.butter(1, 3500.0 / (sr / 2.0), btype='high')
hf_ref = np.mean(np.abs(signal.lfilter(b_air, a_air, clean_ref)))
hf_out = np.mean(np.abs(signal.lfilter(b_air, a_air, out_audio)))

out_path = '/home/usr/Projects/aerovex/modules/sonon/output/speech_synthesis/sonon_arduino_crisp_neural_voice.wav'
sf.write(out_path, out_audio.astype(np.float32), sr)

print(f'Perfected Silk-Crisp Audio Written to: {out_path}')
print(f'  Spikes > 0.8:  {len(spikes_08)}')
print(f'  Spikes > 0.45: {len(spikes_045)}')
print(f'  Max adjacent sample jump: {np.max(diff):.4f}')
print(f'  HF Presence (>3.5kHz): Output={hf_out:.5f}, Target Ref={hf_ref:.5f}')
