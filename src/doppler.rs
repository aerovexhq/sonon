//! Doppler shift compensation, high-speed in-flight kinematic velocity tracking,
//! and relativistic acoustic frequency warping for autonomous UAV speech recognition.

#![deny(unsafe_code)]

use crate::mel::MelFilterbank;

/// Standard speed of sound in dry air at 0 degrees Celsius in m/s.
pub const SPEED_OF_SOUND_0C: f32 = 331.3;

/// Computes the physical speed of sound in air in m/s as a function of temperature in Celsius.
pub fn speed_of_sound_at_temp(temp_celsius: f32) -> f32 {
    let t_kelvin = (temp_celsius + 273.15).max(100.0);
    SPEED_OF_SOUND_0C * (t_kelvin / 273.15).sqrt()
}

/// Configuration parameters for Doppler shift compensation.
#[derive(Debug, Clone, PartialEq)]
pub struct DopplerConfig {
    /// Ambient air temperature in degrees Celsius (typically -20.0 to +45.0 C).
    pub temperature_celsius: f32,
    /// Maximum allowed flight velocity in m/s for safety clamping (e.g. 50.0 m/s = 180 km/h).
    pub max_velocity_mps: f32,
    /// Exponential smoothing alpha for velocity telemetry filtering (0.0 to 1.0).
    pub smoothing_alpha: f32,
    /// Minimum Doppler factor change required to trigger filterbank reconstruction.
    pub filterbank_quantization_step: f32,
}

impl Default for DopplerConfig {
    fn default() -> Self {
        Self {
            temperature_celsius: 20.0,
            max_velocity_mps: 50.0,
            smoothing_alpha: 0.85,
            filterbank_quantization_step: 0.005,
        }
    }
}

/// Dynamic Doppler shift compensator tracking 3D kinematic flight velocity and line-of-sight acoustics.
#[derive(Debug, Clone)]
pub struct DopplerCompensator {
    config: DopplerConfig,
    smoothed_velocity: [f32; 3],
    target_ray: [f32; 3],
    current_doppler_factor: f32,
    cached_filterbank: MelFilterbank,
    cached_factor: f32,
    num_filters: usize,
    fft_size: usize,
    sample_rate: f32,
    base_low_freq: f32,
    base_high_freq: f32,
}

impl DopplerCompensator {
    /// Construct a new Doppler compensator with specified configuration and baseline Mel filterbank parameters.
    pub fn new(
        config: DopplerConfig,
        num_filters: usize,
        fft_size: usize,
        sample_rate: f32,
        low_freq: f32,
        high_freq: f32,
    ) -> Self {
        let base_fb = MelFilterbank::new(num_filters, fft_size, sample_rate, low_freq, high_freq);
        Self {
            config,
            smoothed_velocity: [0.0; 3],
            target_ray: [1.0, 0.0, 0.0], // Default forward boresight ray
            current_doppler_factor: 1.0,
            cached_filterbank: base_fb,
            cached_factor: 1.0,
            num_filters,
            fft_size,
            sample_rate,
            base_low_freq: low_freq,
            base_high_freq: high_freq,
        }
    }

    /// Update drone 3D kinematic velocity vector (vx, vy, vz in m/s, e.g. from MAVLink GLOBAL_POSITION_INT / ODOMETRY).
    pub fn update_velocity_3d(&mut self, vx: f32, vy: f32, vz: f32) {
        let alpha = self.config.smoothing_alpha;
        let v_norm = (vx * vx + vy * vy + vz * vz).sqrt();
        let scale = if v_norm > self.config.max_velocity_mps {
            self.config.max_velocity_mps / v_norm
        } else {
            1.0
        };

        let clamped_vx = vx * scale;
        let clamped_vy = vy * scale;
        let clamped_vz = vz * scale;

        self.smoothed_velocity[0] = alpha * self.smoothed_velocity[0] + (1.0 - alpha) * clamped_vx;
        self.smoothed_velocity[1] = alpha * self.smoothed_velocity[1] + (1.0 - alpha) * clamped_vy;
        self.smoothed_velocity[2] = alpha * self.smoothed_velocity[2] + (1.0 - alpha) * clamped_vz;

        self.recalculate_doppler_factor();
    }

    /// Update target operator line-of-sight unit vector from drone to speaker.
    pub fn update_target_ray(&mut self, rx: f32, ry: f32, rz: f32) {
        let norm = (rx * rx + ry * ry + rz * rz).sqrt();
        if norm > 1e-6 {
            self.target_ray = [rx / norm, ry / norm, rz / norm];
            self.recalculate_doppler_factor();
        }
    }

    /// Update target operator bearing from spherical azimuth and elevation in radians.
    pub fn update_target_bearing(&mut self, azimuth_rad: f32, elevation_rad: f32) {
        let cos_el = elevation_rad.cos();
        let rx = cos_el * azimuth_rad.cos();
        let ry = cos_el * azimuth_rad.sin();
        let rz = elevation_rad.sin();
        self.update_target_ray(rx, ry, rz);
    }

    /// Update ambient temperature telemetry in Celsius.
    pub fn update_temperature(&mut self, temp_celsius: f32) {
        self.config.temperature_celsius = temp_celsius;
        self.recalculate_doppler_factor();
    }

    /// Return the current relativistic acoustic Doppler scaling factor $\alpha = f_{\text{obs}} / f_{\text{src}}$.
    pub fn doppler_factor(&self) -> f32 {
        self.current_doppler_factor
    }

    /// Return active smoothed line-of-sight velocity in m/s (positive = closing, negative = receding).
    pub fn line_of_sight_velocity(&self) -> f32 {
        self.smoothed_velocity[0] * self.target_ray[0]
            + self.smoothed_velocity[1] * self.target_ray[1]
            + self.smoothed_velocity[2] * self.target_ray[2]
    }

    /// Access reference to the cached dynamically tuned Mel filterbank.
    pub fn cached_filterbank(&self) -> &MelFilterbank {
        &self.cached_filterbank
    }

    /// Access mutable or dynamically refreshed reference to active Mel filterbank.
    pub fn active_filterbank(&mut self) -> &MelFilterbank {
        let step = self.config.filterbank_quantization_step;
        let delta = (self.current_doppler_factor - self.cached_factor).abs();

        if delta >= step {
            self.refresh_filterbank();
        }

        &self.cached_filterbank
    }

    /// Reset internal velocity and Doppler state back to stationary baseline.
    pub fn reset(&mut self) {
        self.smoothed_velocity = [0.0; 3];
        self.target_ray = [1.0, 0.0, 0.0];
        self.current_doppler_factor = 1.0;
        self.cached_factor = 1.0;
        self.cached_filterbank = MelFilterbank::new(
            self.num_filters,
            self.fft_size,
            self.sample_rate,
            self.base_low_freq,
            self.base_high_freq,
        );
    }

    /// Fractional cubic Hermite time-domain Doppler resampler.
    /// Resamples an input audio buffer by $1 / \alpha$ to physically undo acoustic pitch and duration shifts.
    pub fn resample_audio(samples: &[f32], doppler_factor: f32) -> Vec<f32> {
        if samples.len() < 4 || (doppler_factor - 1.0).abs() < 1e-4 {
            return samples.to_vec();
        }

        // To restore original pitch from Doppler shifted audio (which was multiplied by alpha),
        // we resample with step = 1.0 / alpha (stretching/compressing back to original duration).
        let step = (1.0 / doppler_factor).clamp(0.5, 2.0);
        let out_len = ((samples.len() as f32) / step).floor() as usize;
        let mut out = Vec::with_capacity(out_len);

        let mut pos = 0.0f32;
        let max_idx = samples.len() - 3;

        while pos < (max_idx as f32) {
            let i = pos.floor() as usize;
            let frac = pos - (i as f32);

            let p0 = samples[i];
            let p1 = samples[i + 1];
            let p2 = samples[i + 2];
            let p3 = samples[i + 3];

            // Cubic Hermite polynomial interpolation
            let c0 = p1;
            let c1 = 0.5 * (p2 - p0);
            let c2 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
            let c3 = 0.5 * (p3 - p0) + 1.5 * (p1 - p2);

            let val = ((c3 * frac + c2) * frac + c1) * frac + c0;
            out.push(val.clamp(-1.0, 1.0));

            pos += step;
        }

        out
    }

    fn refresh_filterbank(&mut self) {
        let alpha = self.current_doppler_factor;
        let nyquist = self.sample_rate / 2.0;

        let shifted_low = (self.base_low_freq * alpha).clamp(20.0, nyquist * 0.4);
        let shifted_high = (self.base_high_freq * alpha).clamp(shifted_low + 500.0, nyquist - 10.0);

        self.cached_filterbank = MelFilterbank::new(
            self.num_filters,
            self.fft_size,
            self.sample_rate,
            shifted_low,
            shifted_high,
        );
        self.cached_factor = alpha;
    }

    fn recalculate_doppler_factor(&mut self) {
        let c = speed_of_sound_at_temp(self.config.temperature_celsius);
        let v_los = self.line_of_sight_velocity();

        // Relativistic acoustic Doppler factor for moving receiver: f_obs = f_src * (1 + v_los / c)
        let alpha = 1.0 + (v_los / c);
        self.current_doppler_factor = alpha.clamp(0.70, 1.30);

        let step = self.config.filterbank_quantization_step;
        let delta = (self.current_doppler_factor - self.cached_factor).abs();
        if delta >= step {
            self.refresh_filterbank();
        }
    }
}
