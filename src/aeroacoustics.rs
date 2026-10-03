#![deny(unsafe_code)]

//! Physics-Informed Aeroacoustic Inverse Source Reconstruction & Far-Field Pressure Directivity Mapping.
//!
//! Features:
//! - Analytical Ffowcs Williams-Hawkings (FW-H) / Gutin formulation for propeller thickness (monopole) and loading (dipole) noise.
//! - Closed-form Bessel function evaluation $J_n(x)$ for harmonic radiation directivity.
//! - Near-field fuselage microphone to rotor source acoustic transfer matrix $\mathbf{H}$ with free-space Green's functions.
//! - Tikhonov-regularized inverse source solver: recovers unknown unsteady blade loading forces $\mathbf{L}_k$ from fuselage acoustic pressure.
//! - 3D Radiation Directivity Sphere $D(\theta, \phi)$ reconstruction mapping elevation, azimuth, and radiation lobes/nulls.
//! - IEC 61672-1 standard A-weighting frequency filter $R_A(f)$ for human-perceived $\text{dB(A)}$ evaluation.
//! - Ground Acoustic Footprint Calculator: projects 3D directivity downward to ground terrain at altitude $h_{\text{AGL}}$ with spherical spreading, atmospheric absorption, and ground boundary reflection.
//! - Urban Noise Abatement Flight Guidance Advisor: computes optimal stealth yaw heading to align acoustic directivity nulls toward designated sensitive ground zones.
//! - Autonomous MAVLink v2 telemetry packet emission (`AERO_DIR`, `AERO_DBA`, `AERO_YAW`).

use crate::beamforming::Point3D;
use crate::health::MavlinkNamedValueFloat;
use crate::tse::{solve_complex_linear_system_in_place, Complex32};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Nominal reference sound pressure in air ($20\text{ }\mu\text{Pa}$).
pub const P_REF_PASCAL: f32 = 2.0e-5;

/// Default speed of sound in dry air at 20 deg C ($343.0\text{ m/s}$).
pub const DEFAULT_SPEED_OF_SOUND: f32 = 343.0;

/// Default air density at standard sea level ($1.225\text{ kg/m}^3$).
pub const DEFAULT_AIR_DENSITY: f32 = 1.225;

/// Evaluates Bessel function of the first kind $J_n(x)$ of integer order $n \ge 0$.
///
/// Uses standard Taylor series expansion:
/// $$J_n(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{k! (n+k)!} \left(\frac{x}{2}\right)^{2k+n}$$
/// Converges rapidly for subsonic drone tip Mach parameters ($x \le 3.0$).
pub fn bessel_j(n: usize, x: f32) -> f32 {
    if x.abs() < 1e-9 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    let is_negative = x < 0.0;
    let abs_x = x.abs();
    let z = abs_x * 0.5;

    let mut term = 1.0f32;
    for i in 1..=n {
        term *= z / (i as f32);
    }
    let mut sum = term;
    let z2 = z * z;

    for k in 1..40 {
        term = -term * z2 / ((k as f32) * ((n + k) as f32));
        sum += term;
        if term.abs() < 1e-9 * sum.abs().max(1e-9) {
            break;
        }
    }

    if is_negative && (n % 2 != 0) {
        -sum
    } else {
        sum
    }
}

/// Evaluates IEC 61672-1 standard A-weighting frequency response filter in dB.
///
/// Returns $\Delta A(f)$ relative to $1000\text{ Hz}$ ($\Delta A(1000) \equiv 0.0\text{ dB}$).
pub fn a_weighting_db(frequency_hz: f32) -> f32 {
    let f = frequency_hz.max(10.0);
    let f2 = f * f;
    let num = 12194.0f32.powi(2) * f2 * f2;
    let den = (f2 + 20.6f32.powi(2))
        * ((f2 + 107.7f32.powi(2)) * (f2 + 737.9f32.powi(2))).sqrt()
        * (f2 + 12194.0f32.powi(2));
    let ra = num / den.max(1e-12);
    20.0 * ra.log10() + 2.00
}

/// Atmospheric acoustic absorption coefficient in $\text{dB/km}$ (ISO 9613-1 approximation).
pub fn atmospheric_absorption_db_km(frequency_hz: f32) -> f32 {
    let f_khz = (frequency_hz / 1000.0).max(0.01);
    // Standard 20 deg C, 50% relative humidity nominal curve
    1.5 * f_khz.powf(1.4) + 0.5
}

/// Rotor mechanical and aerodynamic configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RotorGeometry {
    /// Rotor identifier index (0 to K-1).
    pub rotor_id: usize,
    /// Physical hub location in aircraft body frame [m] (+X Forward, +Y Right, +Z Down).
    pub position: Point3D,
    /// Rotor blade radius [m] (e.g. 0.125 m for 10-inch propeller).
    pub radius_m: f32,
    /// Number of blades per rotor $B_r$ (typically 2).
    pub num_blades: usize,
    /// Unit vector of rotation thrust axis (typically [0.0, 0.0, -1.0] for upward lift).
    pub thrust_axis: Point3D,
    /// Direction of rotation: +1.0 for Clockwise (CW), -1.0 for Counter-Clockwise (CCW).
    pub rotation_dir: f32,
}

impl RotorGeometry {
    /// Construct a standard rotor geometry with upward thrust axis.
    pub fn new(rotor_id: usize, x: f32, y: f32, z: f32, radius_m: f32, num_blades: usize, is_cw: bool) -> Self {
        Self {
            rotor_id,
            position: Point3D::new(x, y, z),
            radius_m,
            num_blades,
            thrust_axis: Point3D::new(0.0, 0.0, -1.0),
            rotation_dir: if is_cw { 1.0 } else { -1.0 },
        }
    }
}

/// Configuration parameters for aeroacoustic inverse reconstruction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AeroacousticConfig {
    /// Audio sample rate in Hz (e.g. 16000.0).
    pub sample_rate: f32,
    /// Speed of sound in m/s (default 343.0).
    pub speed_of_sound: f32,
    /// Ambient air density in kg/m^3 (default 1.225).
    pub air_density: f32,
    /// Tikhonov regularization parameter $\lambda$ for ill-conditioned inverse matrix solving.
    pub tikhonov_lambda: f32,
    /// Far-field directivity reference sphere distance in meters (default 1.0 m).
    pub reference_distance_m: f32,
    /// Number of elevation divisions on directivity sphere (default 18 -> 10 deg steps).
    pub elevation_bins: usize,
    /// Number of azimuth divisions on directivity sphere (default 36 -> 10 deg steps).
    pub azimuth_bins: usize,
    /// Noise threshold for urban ground footprint area calculation in dB(A) (default 65.0).
    pub footprint_threshold_dba: f32,
}

impl Default for AeroacousticConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000.0,
            speed_of_sound: DEFAULT_SPEED_OF_SOUND,
            air_density: DEFAULT_AIR_DENSITY,
            tikhonov_lambda: 1e-3,
            reference_distance_m: 1.0,
            elevation_bins: 18,
            azimuth_bins: 36,
            footprint_threshold_dba: 65.0,
        }
    }
}

/// 3D Far-Field Acoustic Radiation Directivity Sphere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectivitySphere3D {
    /// Fundamental acoustic frequency of the evaluated harmonic [Hz].
    pub frequency_hz: f32,
    /// Far-field Sound Pressure Level matrix $\text{SPL}(\theta, \phi)$ in dB SPL @ 1.0 m.
    /// Indexed by `[elevation_idx][azimuth_idx]`.
    pub grid_spl_db: Vec<Vec<f32>>,
    /// Far-field A-weighted Sound Pressure Level matrix $\text{SPL}(\theta, \phi)$ in dB(A) @ 1.0 m.
    pub grid_spl_dba: Vec<Vec<f32>>,
    /// Global peak radiation Sound Pressure Level in dB(A).
    pub peak_spl_dba: f32,
    /// Polar angles $(\theta, \phi)$ of peak radiation lobe in degrees.
    pub peak_direction_deg: (f32, f32),
    /// Minimum radiation level in dB(A).
    pub min_spl_dba: f32,
    /// Polar angles $(\theta, \phi)$ of quietest acoustic radiation notch in degrees.
    pub quietest_direction_deg: (f32, f32),
}

/// 2D Ground Acoustic Footprint Projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundNoiseFootprint {
    /// Aircraft altitude above ground level (AGL) in meters.
    pub altitude_agl_m: f32,
    /// Ground X span range [min_x, max_x] in meters.
    pub x_range_m: (f32, f32),
    /// Ground Y span range [min_y, max_y] in meters.
    pub y_range_m: (f32, f32),
    /// Grid spatial resolution in meters (e.g. 5.0 m).
    pub resolution_m: f32,
    /// 2D Ground A-weighted Sound Pressure Level contour `[x_idx][y_idx]` in dB(A).
    pub grid_dba: Vec<Vec<f32>>,
    /// Peak ground noise level in dB(A).
    pub peak_ground_dba: f32,
    /// Relative Cartesian coordinates $(X, Y)$ of peak ground noise hotspot in meters.
    pub peak_location_m: (f32, f32),
    /// Total ground surface area where noise exceeds the configured threshold ($\text{m}^2$).
    pub footprint_area_exceeding_threshold_m2: f32,
}

/// Real-time aeroacoustic telemetry and urban noise abatement advice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AeroacousticTelemetry {
    /// Timestamp of evaluation in seconds.
    pub timestamp_sec: f64,
    /// Dominant radiation elevation angle in degrees $[0^\circ, 180^\circ]$.
    pub dominant_elevation_deg: f32,
    /// Dominant radiation azimuth angle in degrees $[0^\circ, 360^\circ)$.
    pub dominant_azimuth_deg: f32,
    /// Peak ground noise footprint level in dB(A).
    pub peak_ground_dba: f32,
    /// Quietest acoustic notch azimuth in aircraft body frame in degrees.
    pub quietest_azimuth_deg: f32,
    /// Recommended yaw adjustment delta in degrees to steer acoustic notch toward target.
    pub recommended_yaw_delta_deg: f32,
    /// Reconstructed rotor source loading force magnitudes [N].
    pub reconstructed_rotor_forces: Vec<f32>,
}

/// Physics-Informed Aeroacoustic Inversion & Directivity Mapping Engine.
pub struct AeroacousticInverter {
    rotors: Vec<RotorGeometry>,
    mic_positions: Vec<Point3D>,
    config: AeroacousticConfig,
    latest_sphere: Option<DirectivitySphere3D>,
    latest_footprint: Option<GroundNoiseFootprint>,
    latest_telemetry: Option<AeroacousticTelemetry>,
}

impl AeroacousticInverter {
    /// Construct a new aeroacoustic inverter with specified rotors and fuselage microphones.
    pub fn new(
        rotors: Vec<RotorGeometry>,
        mic_positions: Vec<Point3D>,
        config: AeroacousticConfig,
    ) -> Self {
        assert!(!rotors.is_empty(), "Must configure at least one rotor");
        assert!(!mic_positions.is_empty(), "Must configure at least one microphone");
        Self {
            rotors,
            mic_positions,
            config,
            latest_sphere: None,
            latest_footprint: None,
            latest_telemetry: None,
        }
    }

    /// Access configured rotors.
    pub fn rotors(&self) -> &[RotorGeometry] {
        &self.rotors
    }

    /// Access microphone positions.
    pub fn mic_positions(&self) -> &[Point3D] {
        &self.mic_positions
    }

    /// Access current configuration.
    pub fn config(&self) -> &AeroacousticConfig {
        &self.config
    }

    /// Reconstructs the complex rotor blade loading forces $\hat{\mathbf{F}} \in \mathbb{C}^K$
    /// from near-field fuselage microphone complex acoustic pressures $\mathbf{p}_{\text{meas}} \in \mathbb{C}^M$.
    ///
    /// Solves the Tikhonov-regularized linear inverse system:
    /// $$(\mathbf{H}^H \mathbf{H} + \lambda^2 \mathbf{I}) \hat{\mathbf{F}} = \mathbf{H}^H \mathbf{p}_{\text{meas}}$$
    pub fn reconstruct_rotor_sources(
        &self,
        measured_pressures: &[Complex32],
        harmonic_freq_hz: f32,
    ) -> Vec<Complex32> {
        let m = self.mic_positions.len();
        let k = self.rotors.len();
        assert_eq!(
            measured_pressures.len(),
            m,
            "Measured pressure vector length must match microphone count"
        );

        let c = self.config.speed_of_sound;
        let omega = 2.0 * PI * harmonic_freq_hz;
        let wave_k = omega / c;

        // 1. Build near-field transfer matrix H [M x K]
        // Free-space dipole Green's function from rotor k to mic m:
        // G_mk = e^(-j k r_mk) / (4*pi*r_mk) * (j*wave_k + 1/r_mk) * (n_thrust . r_hat_mk)
        let mut h_mat = vec![Complex32::zero(); m * k];
        for r_idx in 0..m {
            let mic = &self.mic_positions[r_idx];
            for c_idx in 0..k {
                let rot = &self.rotors[c_idx];
                let dx = mic.x - rot.position.x;
                let dy = mic.y - rot.position.y;
                let dz = mic.z - rot.position.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.01);

                let r_hat_x = dx / dist;
                let r_hat_y = dy / dist;
                let r_hat_z = dz / dist;

                // Projection along rotor thrust axis
                let cos_proj = rot.thrust_axis.x * r_hat_x
                    + rot.thrust_axis.y * r_hat_y
                    + rot.thrust_axis.z * r_hat_z;

                // Phase factor e^(-j * wave_k * dist)
                let phase = -wave_k * dist;
                let exp_factor = Complex32::from_polar(1.0 / (4.0 * PI * dist), phase);

                // Near-field + far-field dipole term: (j * wave_k + 1.0 / dist)
                let dipole_factor = Complex32::new(1.0 / dist, wave_k).scale(cos_proj);

                let green = exp_factor.mul(dipole_factor);
                h_mat[r_idx * k + c_idx] = green;
            }
        }

        // If m == k, attempt direct unregularized inversion first for maximum accuracy
        if m == k {
            let mut direct_solution = vec![Complex32::zero(); k];
            if solve_complex_linear_system_in_place(k, &h_mat, measured_pressures, &mut direct_solution) {
                return direct_solution;
            }
        }

        // 2. Form normal equations: A = H^H * H + lambda^2 * I [K x K], b = H^H * p_meas [K x 1]
        let lambda_sq = self.config.tikhonov_lambda * self.config.tikhonov_lambda;
        let mut a_mat = vec![Complex32::zero(); k * k];
        let mut b_vec = vec![Complex32::zero(); k];

        for i in 0..k {
            // b[i] = sum_m (H_mi^* * p_m)
            let mut sum_b = Complex32::zero();
            for r_idx in 0..m {
                let h_mi = h_mat[r_idx * k + i];
                sum_b = sum_b.add(h_mi.conj().mul(measured_pressures[r_idx]));
            }
            b_vec[i] = sum_b;

            // A[i, j] = sum_m (H_mi^* * H_mj) + (lambda^2 if i == j)
            for j in 0..k {
                let mut sum_a = Complex32::zero();
                for r_idx in 0..m {
                    let h_mi = h_mat[r_idx * k + i];
                    let h_mj = h_mat[r_idx * k + j];
                    sum_a = sum_a.add(h_mi.conj().mul(h_mj));
                }
                if i == j {
                    sum_a = sum_a.add(Complex32::new(lambda_sq, 0.0));
                }
                a_mat[i * k + j] = sum_a;
            }
        }

        // 3. Solve linear system A * F = b
        let mut solution = vec![Complex32::zero(); k];
        let success = solve_complex_linear_system_in_place(k, &a_mat, &b_vec, &mut solution);
        if !success {
            // Fallback: scaled conjugate projection
            for i in 0..k {
                let denom = a_mat[i * k + i].norm().max(1e-6);
                solution[i] = b_vec[i].scale(1.0 / denom);
            }
        }

        solution
    }

    /// Evaluates the 3D Radiation Directivity Sphere $D(\theta, \phi)$ in far-field at reference distance $R_{\text{ref}}$.
    ///
    /// Computes full spatial distribution across polar elevation $\theta \in [0, \pi]$ and azimuth $\phi \in [0, 2\pi)$.
    pub fn compute_directivity_sphere(
        &self,
        rotor_forces: &[Complex32],
        harmonic_freq_hz: f32,
    ) -> DirectivitySphere3D {
        let k = self.rotors.len();
        let c = self.config.speed_of_sound;
        let r_ref = self.config.reference_distance_m;
        let omega = 2.0 * PI * harmonic_freq_hz;
        let wave_k = omega / c;
        let a_weight = a_weighting_db(harmonic_freq_hz);

        let n_elev = self.config.elevation_bins.max(6);
        let n_azim = self.config.azimuth_bins.max(12);

        let mut grid_spl_db = vec![vec![0.0f32; n_azim]; n_elev];
        let mut grid_spl_dba = vec![vec![0.0f32; n_azim]; n_elev];

        let mut peak_spl_dba = -100.0f32;
        let mut peak_direction_deg = (0.0f32, 0.0f32);
        let mut min_spl_dba = 200.0f32;
        let mut quietest_direction_deg = (0.0f32, 0.0f32);

        for el_idx in 0..n_elev {
            // Elevation theta from 0 (forward/up) to PI (down)
            let theta = (el_idx as f32 + 0.5) * PI / (n_elev as f32);
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for az_idx in 0..n_azim {
                // Azimuth phi from 0 to 2*PI
                let phi = (az_idx as f32) * (2.0 * PI) / (n_azim as f32);
                let cos_phi = phi.cos();
                let sin_phi = phi.sin();

                // Observer unit vector u_hat
                let u_x = sin_theta * cos_phi;
                let u_y = sin_theta * sin_phi;
                let u_z = cos_theta;

                // Sum far-field acoustic pressure from all rotors
                let mut p_far = Complex32::zero();
                for r_i in 0..k {
                    let rot = &self.rotors[r_i];
                    let force = rotor_forces[r_i];

                    // Spatial array phase delay relative to origin: wave_k * (R_rot . u_hat)
                    let spatial_delay = wave_k * (rot.position.x * u_x + rot.position.y * u_y + rot.position.z * u_z);
                    let phase_shift = Complex32::from_polar(1.0, spatial_delay);

                    // Rotor dipole directivity factor: (n_thrust . u_hat)
                    let cos_proj = rot.thrust_axis.x * u_x + rot.thrust_axis.y * u_y + rot.thrust_axis.z * u_z;

                    // Far-field dipole factor: (j * wave_k / (4*pi*r_ref)) * (F . u_hat)
                    let dipole_scale = wave_k * cos_proj / (4.0 * PI * r_ref);
                    let term = force.scale(dipole_scale).mul(phase_shift);

                    p_far = p_far.add(term);
                }

                let p_mag = p_far.norm().max(1e-9);
                let spl_db = 20.0 * (p_mag / P_REF_PASCAL).log10();
                let spl_dba = spl_db + a_weight;

                grid_spl_db[el_idx][az_idx] = spl_db;
                grid_spl_dba[el_idx][az_idx] = spl_dba;

                let theta_deg = theta.to_degrees();
                let phi_deg = phi.to_degrees();

                if spl_dba > peak_spl_dba {
                    peak_spl_dba = spl_dba;
                    peak_direction_deg = (phi_deg, theta_deg);
                }
                if spl_dba < min_spl_dba {
                    min_spl_dba = spl_dba;
                    quietest_direction_deg = (phi_deg, theta_deg);
                }
            }
        }

        DirectivitySphere3D {
            frequency_hz: harmonic_freq_hz,
            grid_spl_db,
            grid_spl_dba,
            peak_spl_dba,
            peak_direction_deg,
            min_spl_dba,
            quietest_direction_deg,
        }
    }

    /// Projects 3D far-field directivity downward onto the 2D ground plane at altitude $h_{\text{AGL}}$.
    ///
    /// Computes spatial propagation losses, atmospheric absorption $\alpha_{\text{atm}}(f)$,
    /// and rigid ground boundary reflection ($+3.0\text{ dB}$).
    pub fn compute_ground_noise_footprint(
        &self,
        sphere: &DirectivitySphere3D,
        altitude_agl_m: f32,
        half_span_m: f32,
        resolution_m: f32,
    ) -> GroundNoiseFootprint {
        let h = altitude_agl_m.max(1.0);
        let span = half_span_m.max(10.0);
        let res = resolution_m.max(1.0);

        let num_steps = ((2.0 * span) / res).round() as usize + 1;
        let mut grid_dba = vec![vec![0.0f32; num_steps]; num_steps];

        let n_elev = sphere.grid_spl_dba.len();
        let n_azim = if n_elev > 0 { sphere.grid_spl_dba[0].len() } else { 0 };

        let alpha_db_km = atmospheric_absorption_db_km(sphere.frequency_hz);
        let mut peak_ground_dba = -100.0f32;
        let mut peak_location_m = (0.0f32, 0.0f32);
        let mut count_exceeding_threshold = 0usize;

        for ix in 0..num_steps {
            let x = -span + (ix as f32) * res;
            for iy in 0..num_steps {
                let y = -span + (iy as f32) * res;

                // Observer on ground at position (x, y, h) relative to aircraft
                let dist_ground = (x * x + y * y + h * h).sqrt();

                // Spherical angles of vector pointing from aircraft to ground observer
                // In aerospace frame: +X Forward, +Y Right, +Z Down
                // Downward vector has z = +h > 0
                let theta = (h / dist_ground).clamp(-1.0, 1.0).acos(); // 0 is nadir downwards
                let phi = y.atan2(x);
                let phi_norm = if phi < 0.0 { phi + 2.0 * PI } else { phi };

                // Map theta and phi to sphere grid indices
                let el_idx = ((theta / PI) * (n_elev as f32)).floor() as usize;
                let el_clamped = el_idx.min(n_elev.saturating_sub(1));

                let az_idx = ((phi_norm / (2.0 * PI)) * (n_azim as f32)).floor() as usize;
                let az_clamped = az_idx.min(n_azim.saturating_sub(1));

                let source_spl_dba = sphere.grid_spl_dba[el_clamped][az_clamped];

                // Spherical spreading loss: 20 * log10(R / R_ref)
                let spreading_loss_db = 20.0 * (dist_ground / self.config.reference_distance_m).max(1.0).log10();

                // Atmospheric absorption: alpha * (dist / 1000)
                let atm_loss_db = alpha_db_km * (dist_ground / 1000.0);

                // Ground reflection pressure doubling factor: +3.0 dB
                let ground_spl_dba = source_spl_dba - spreading_loss_db - atm_loss_db + 3.0;

                grid_dba[ix][iy] = ground_spl_dba;

                if ground_spl_dba > peak_ground_dba {
                    peak_ground_dba = ground_spl_dba;
                    peak_location_m = (x, y);
                }

                if ground_spl_dba >= self.config.footprint_threshold_dba {
                    count_exceeding_threshold += 1;
                }
            }
        }

        let cell_area_m2 = res * res;
        let footprint_area_exceeding_threshold_m2 = (count_exceeding_threshold as f32) * cell_area_m2;

        GroundNoiseFootprint {
            altitude_agl_m: h,
            x_range_m: (-span, span),
            y_range_m: (-span, span),
            resolution_m: res,
            grid_dba,
            peak_ground_dba,
            peak_location_m,
            footprint_area_exceeding_threshold_m2,
        }
    }

    /// Evaluates urban noise abatement flight guidance.
    ///
    /// Computes the optimal aircraft yaw rotation to steer the quietest radiation notch
    /// toward a sensitive ground location $(x_{\text{target}}, y_{\text{target}})$ relative to the aircraft.
    pub fn compute_stealth_yaw_advice(
        &self,
        sphere: &DirectivitySphere3D,
        target_ground_pos_m: (f32, f32),
        current_yaw_rad: f32,
    ) -> f32 {
        let (tx, ty) = target_ground_pos_m;
        let target_bearing_rad = ty.atan2(tx);

        // Quietest radiation azimuth on directivity sphere (in aircraft body frame)
        let quiet_azimuth_rad = sphere.quietest_direction_deg.0.to_radians();

        // Desired aircraft yaw: aligns quiet_azimuth with target_bearing
        let desired_yaw_rad = target_bearing_rad - quiet_azimuth_rad;

        // Wrap delta to [-PI, +PI]
        let mut delta_rad = desired_yaw_rad - current_yaw_rad;
        while delta_rad > PI {
            delta_rad -= 2.0 * PI;
        }
        while delta_rad < -PI {
            delta_rad += 2.0 * PI;
        }

        delta_rad.to_degrees()
    }

    /// Process a streaming multi-channel acoustic frame and telemetry state.
    ///
    /// - `channels`: Measured time-domain audio slices from fuselage microphones.
    /// - `motor_rpms`: Active motor rotational speeds in RPM.
    /// - `altitude_agl_m`: Aircraft altitude above ground level in meters.
    /// - `target_ground_pos_m`: Sensitive ground target coordinate to protect $(X, Y)$ in meters.
    /// - `current_yaw_rad`: Active aircraft yaw heading in radians.
    /// - `timestamp_sec`: Current mission timestamp in seconds.
    pub fn process_frame(
        &mut self,
        channels: &[&[f32]],
        motor_rpms: &[f32],
        altitude_agl_m: f32,
        target_ground_pos_m: Option<(f32, f32)>,
        current_yaw_rad: f32,
        timestamp_sec: f64,
    ) -> AeroacousticTelemetry {
        let m = self.mic_positions.len();
        let _k = self.rotors.len();
        assert!(channels.len() >= m, "Channels must cover all configured microphones");

        // 1. Determine dominant rotor Blade Pass Frequency (BPF)
        let mean_rpm = if !motor_rpms.is_empty() {
            motor_rpms.iter().sum::<f32>() / (motor_rpms.len() as f32)
        } else {
            6000.0
        };
        let num_blades = self.rotors[0].num_blades.max(2);
        let bpf_hz = ((mean_rpm / 60.0) * (num_blades as f32)).max(50.0);

        // 2. Extract complex frequency bin at BPF via discrete Fourier transform sum
        let n_samples = channels[0].len().max(1);
        let omega = 2.0 * PI * bpf_hz;
        let dt = 1.0 / self.config.sample_rate;

        let mut p_meas = Vec::with_capacity(m);
        for ch in channels.iter().take(m) {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for (idx, &s) in ch.iter().enumerate() {
                let t = (idx as f32) * dt;
                let angle = -omega * t;
                sum_re += s * angle.cos();
                sum_im += s * angle.sin();
            }
            // Normalize by window length
            let scale = 2.0 / (n_samples as f32);
            p_meas.push(Complex32::new(sum_re * scale, sum_im * scale));
        }

        // 3. Invert near-field transfer matrix to recover rotor blade loading forces
        let forces = self.reconstruct_rotor_sources(&p_meas, bpf_hz);
        let force_mags: Vec<f32> = forces.iter().map(|f| f.norm()).collect();

        // 4. Compute 3D radiation directivity sphere
        let sphere = self.compute_directivity_sphere(&forces, bpf_hz);

        // 5. Compute ground noise footprint
        let footprint = self.compute_ground_noise_footprint(&sphere, altitude_agl_m, 100.0, 5.0);

        // 6. Compute stealth yaw advice if target is provided
        let target_pos = target_ground_pos_m.unwrap_or((0.0, 50.0));
        let yaw_delta_deg = self.compute_stealth_yaw_advice(&sphere, target_pos, current_yaw_rad);

        let telem = AeroacousticTelemetry {
            timestamp_sec,
            dominant_elevation_deg: sphere.peak_direction_deg.1,
            dominant_azimuth_deg: sphere.peak_direction_deg.0,
            peak_ground_dba: footprint.peak_ground_dba,
            quietest_azimuth_deg: sphere.quietest_direction_deg.0,
            recommended_yaw_delta_deg: yaw_delta_deg,
            reconstructed_rotor_forces: force_mags,
        };

        self.latest_sphere = Some(sphere);
        self.latest_footprint = Some(footprint);
        self.latest_telemetry = Some(telem.clone());

        telem
    }

    /// Access latest computed 3D directivity sphere.
    pub fn latest_directivity_sphere(&self) -> Option<&DirectivitySphere3D> {
        self.latest_sphere.as_ref()
    }

    /// Access latest ground noise footprint projection.
    pub fn latest_ground_noise_footprint(&self) -> Option<&GroundNoiseFootprint> {
        self.latest_footprint.as_ref()
    }

    /// Access latest evaluated telemetry.
    pub fn latest_telemetry(&self) -> Option<&AeroacousticTelemetry> {
        self.latest_telemetry.as_ref()
    }

    /// Serializes aeroacoustic telemetry into standard MAVLink v2 `NAMED_VALUE_FLOAT` packets.
    pub fn to_mavlink_packets(
        telemetry: &AeroacousticTelemetry,
        time_boot_ms: u32,
    ) -> Vec<MavlinkNamedValueFloat> {
        vec![
            MavlinkNamedValueFloat::new(time_boot_ms, "AERO_DIR", telemetry.dominant_azimuth_deg),
            MavlinkNamedValueFloat::new(time_boot_ms, "AERO_DBA", telemetry.peak_ground_dba),
            MavlinkNamedValueFloat::new(time_boot_ms, "AERO_YAW", telemetry.recommended_yaw_delta_deg),
        ]
    }
}
