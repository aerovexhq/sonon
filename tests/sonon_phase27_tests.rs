#![deny(unsafe_code)]

//! Test Suite for Sonon Phase 27: Distributed Multi-UAV Swarm Acoustic Mesh Beamforming
//! & Synthetic Aperture Acoustic Radar.

use sonon::beamforming::Point3D;
use sonon::engine::SononEngine;
use sonon::swarm_mesh::{
    SwarmClockSync, SwarmCovarianceConsensus, SwarmMeshConfig, SwarmNodeState,
    SyntheticApertureBeamformer,
};
use sonon::tse::Complex32;
use std::f32::consts::PI;
use std::time::Instant;

/// Helper to configure a 4-drone distributed acoustic swarm across a 60-meter baseline aperture.
fn setup_4_drone_distributed_swarm() -> (SyntheticApertureBeamformer, Vec<SwarmNodeState>) {
    let config = SwarmMeshConfig {
        sample_rate: 16000.0,
        speed_of_sound: 343.0,
        target_freq_hz: 180.0, // 180 Hz vehicle diesel engine tonal harmonic
        search_bandwidth_hz: 40.0,
        max_baseline_m: 80.0,
        consensus_iterations: 5,
        angular_step_deg: 0.25, // 0.25 deg high-precision scanning
        min_detection_snr_db: 3.0,
    };

    let mut beamformer = SyntheticApertureBeamformer::new(0, config);

    // 4 UAVs in diamond formation at altitude 25 meters (Z = -25 m in NED frame)
    // Span: 60 meters from West to East, 60 meters from North to South
    let nodes = vec![
        SwarmNodeState::new(0, 30.0, 0.0, -25.0),   // Node 0: North (+X)
        SwarmNodeState::new(1, 0.0, 30.0, -25.0),   // Node 1: East (+Y)
        SwarmNodeState::new(2, -30.0, 0.0, -25.0),  // Node 2: South (-X)
        SwarmNodeState::new(3, 0.0, -30.0, -25.0),  // Node 3: West (-Y)
    ];

    beamformer.update_swarm_nodes(nodes.clone());
    (beamformer, nodes)
}

/// Test 1: IEEE 802.15.4z UWB Sub-Microsecond Clock Synchronization Model.
#[test]
fn test_uwb_sub_microsecond_clock_synchronization() {
    let mut clock_sync = SwarmClockSync::new(0.1e-6);

    // True physical values:
    // Distance between Master (Node 0) and Node 1: 42.42 meters -> true ToF = 42.42 / 3e8 = 1.414e-7 sec (141.4 ns)
    let true_tof = 1.414e-7;
    let true_offset = 2.45e-6; // 2.45 microseconds clock offset

    // Simulate 10 consecutive Two-Way Ranging (TWR) packet exchanges
    for step in 0..10 {
        let t_base = (step as f64) * 0.1; // 100 ms interval
        let t1 = t_base;
        let t2 = t1 + true_tof + true_offset;
        let t3 = t2 + 0.001; // 1.0 ms turnaround time
        let t4 = t3 + true_tof - true_offset;

        let measured_tof = clock_sync.process_twr_exchange(t1, t2, t3, t4);
        assert!((measured_tof - true_tof).abs() < 1e-9, "ToF estimation error must be < 1 ns");
    }

    let offset_err = (clock_sync.estimated_offset_sec - true_offset).abs();
    println!("Estimated clock offset: {:.4} us, Truth: {:.4} us, Error: {:.6} us",
        clock_sync.estimated_offset_sec * 1e6, true_offset * 1e6, offset_err * 1e6
    );

    // Sub-microsecond accuracy verification:
    assert!(
        offset_err < 0.1e-6,
        "Clock offset estimation error must be < 0.1 us (100 ns), got: {:.3} us",
        offset_err * 1e6
    );

    // Synchronized timestamp conversion:
    let local_ts = 10.00000245;
    let sync_ts = clock_sync.to_synchronized_time(local_ts);
    assert!((sync_ts - 10.0).abs() < 1e-7);
}

/// Test 2: Distributed Synthetic Aperture Acoustic Radar Sub-Degree Angular Resolution.
#[test]
fn test_synthetic_aperture_sub_degree_angular_resolution() {
    let (beamformer, nodes) = setup_4_drone_distributed_swarm();

    let baseline = beamformer.calculate_aperture_baseline();
    println!("Swarm synthetic aperture baseline: {:.1} meters", baseline);
    assert!(baseline >= 55.0, "Aperture baseline must exceed 50 meters, got: {baseline:.1} m");

    // Target acoustic source on ground (Z = 0) at Azimuth theta = 55.0 degrees, range = 250 meters
    let target_azimuth_deg = 55.0f32;
    let target_range_m = 250.0f32;
    let target_azimuth_rad = target_azimuth_deg.to_radians();

    let target_x = target_range_m * target_azimuth_rad.cos();
    let target_y = target_range_m * target_azimuth_rad.sin();
    let target_z = 0.0f32; // on ground

    let target_pos = Point3D::new(target_x, target_y, target_z);

    // Calculate acoustic signal received at each of the 4 swarm nodes
    let c = 343.0f32;
    let freq = 180.0f32;
    let omega = 2.0 * PI * freq;
    let wave_k = omega / c;

    let mut node_signals = Vec::with_capacity(nodes.len());
    for node in &nodes {
        let dx = node.position.x - target_pos.x;
        let dy = node.position.y - target_pos.y;
        let dz = node.position.z - target_pos.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        // Spherical wave arrival: exp(-j * k * dist) / dist
        let phase = -wave_k * dist;
        let amp = 100.0 / dist; // 1/R geometric spreading
        node_signals.push(Complex32::from_polar(amp, phase));
    }

    // Locate target using distributed synthetic aperture beamforming
    let report = beamformer
        .locate_ground_target(&node_signals, 1.234)
        .expect("Swarm target localization must succeed");

    let az_error = (report.azimuth_deg - target_azimuth_deg).abs();
    println!(
        "Detected Azimuth: {:.2} deg, Ground Truth: {:.2} deg, Error: {:.3} deg",
        report.azimuth_deg, target_azimuth_deg, az_error
    );

    // Sub-degree angular precision requirement (< 0.5 degrees):
    assert!(
        az_error < 0.50,
        "Distributed swarm azimuth bearing error must be < 0.50 degrees, got: {az_error:.3} deg"
    );

    // Coherent array gain verification: 10 * log10(4) = 6.02 dB
    assert!(
        (report.array_gain_db - 6.02).abs() < 0.1,
        "4-drone array gain must equal ~6.02 dB, got: {:.2} dB",
        report.array_gain_db
    );
}

/// Test 3: Decentralized Spatial Covariance Consensus Convergence.
#[test]
fn test_decentralized_covariance_consensus_convergence() {
    let num_nodes = 4;
    let mut consensus_nodes: Vec<SwarmCovarianceConsensus> = (0..num_nodes)
        .map(|id| SwarmCovarianceConsensus::new(id, num_nodes))
        .collect();

    // Each node observes a local covariance matrix with unique local noise perturbations
    let mut ground_truth_mean = vec![Complex32::zero(); num_nodes * num_nodes];

    for (id, node) in consensus_nodes.iter_mut().enumerate() {
        let mut local_cov = vec![Complex32::zero(); num_nodes * num_nodes];
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                let val = Complex32::new(
                    (i + j + 1) as f32 + (id as f32) * 0.5,
                    ((i as f32) - (j as f32)) * 0.2,
                );
                local_cov[i * num_nodes + j] = val;
                ground_truth_mean[i * num_nodes + j] = ground_truth_mean[i * num_nodes + j].add(val);
            }
        }
        node.set_local_covariance(&local_cov);
    }

    // Exact ground truth average
    let inv_n = 1.0 / (num_nodes as f32);
    for v in &mut ground_truth_mean {
        *v = v.scale(inv_n);
    }

    // Ring topology: Node 0 <-> 1 <-> 2 <-> 3 <-> 0 (each node has degree 2)
    let neighbors = [
        vec![1, 3], // Node 0
        vec![0, 2], // Node 1
        vec![1, 3], // Node 2
        vec![2, 0], // Node 3
    ];

    // Execute 6 consensus iterations
    for _ in 0..6 {
        // Collect current states
        let current_states: Vec<Vec<Complex32>> = consensus_nodes
            .iter()
            .map(|n| n.consensus_covariance().to_vec())
            .collect();

        for i in 0..num_nodes {
            let neigh_covs: Vec<&[Complex32]> = neighbors[i]
                .iter()
                .map(|&idx| &current_states[idx][..])
                .collect();
            let neigh_degs = vec![2, 2];

            consensus_nodes[i].step_consensus(&neigh_covs, &neigh_degs, 2);
        }
    }

    // Verify consensus convergence to global mean across all nodes
    for (id, node) in consensus_nodes.iter().enumerate() {
        let result = node.consensus_covariance();
        let mut max_err = 0.0f32;
        for k in 0..(num_nodes * num_nodes) {
            let err = result[k].sub(ground_truth_mean[k]).norm();
            if err > max_err {
                max_err = err;
            }
        }
        println!("Node {id} max consensus error: {:.5}", max_err);
        assert!(
            max_err < 0.15,
            "Node {id} consensus must converge to global mean, max error: {max_err:.5}"
        );
    }
}

/// Test 4: Spherical Wavefront Fresnel Ranging & 3D Ground Vehicle Localization.
#[test]
fn test_spherical_wavefront_fresnel_ranging_and_localization() {
    let (beamformer, nodes) = setup_4_drone_distributed_swarm();

    // Target vehicle on ground at Azimuth = 90 deg (East, +Y = 100 m, X = 0, Z = 0)
    let truth_range = 100.0f32;
    let target_pos = Point3D::new(0.0, truth_range, 0.0);

    let c = 343.0f32;
    let freq = 180.0f32;
    let omega = 2.0 * PI * freq;
    let wave_k = omega / c;

    let mut node_signals = Vec::with_capacity(nodes.len());
    for node in &nodes {
        let dx = node.position.x - target_pos.x;
        let dy = node.position.y - target_pos.y;
        let dz = node.position.z - target_pos.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        let phase = -wave_k * dist;
        node_signals.push(Complex32::from_polar(1.0, phase));
    }

    let report = beamformer
        .locate_ground_target(&node_signals, 0.0)
        .expect("Target localization must succeed");

    println!(
        "Estimated 3D position: ({:.1}, {:.1}, {:.1}) m, Truth: ({:.1}, {:.1}, {:.1}) m, Range: {:.1} m",
        report.position.x, report.position.y, report.position.z,
        target_pos.x, target_pos.y, target_pos.z,
        report.range_m
    );

    // Bearing verification: East corresponds to Azimuth 90 degrees
    let az_err = (report.azimuth_deg - 90.0).abs();
    assert!(az_err < 1.0, "Azimuth must be near 90 deg, got: {:.2} deg", report.azimuth_deg);

    // Range bracket verification (100 meters target)
    assert!(
        (report.range_m - truth_range).abs() < 30.0,
        "Estimated range must be within candidate bracket, got: {:.1} m",
        report.range_m
    );
}

/// Test 5: Integrated Sonon Engine Swarm Mesh Processing and MAVLink Telemetry Throughput.
#[test]
fn test_integrated_sonon_engine_swarm_mesh_throughput() {
    let (beamformer, nodes) = setup_4_drone_distributed_swarm();
    let config = beamformer.nodes().first().map(|_| SwarmMeshConfig {
        sample_rate: 16000.0,
        speed_of_sound: 343.0,
        target_freq_hz: 180.0,
        search_bandwidth_hz: 40.0,
        max_baseline_m: 80.0,
        consensus_iterations: 5,
        angular_step_deg: 0.5,
        min_detection_snr_db: 3.0,
    }).unwrap();

    let mut engine = SononEngine::new(16000.0, 512, 256, 13);
    engine.enable_swarm_mesh_beamforming(0, config);
    engine.update_swarm_nodes(nodes.clone());

    assert!(engine.swarm_beamformer().is_some());

    // Synthesize audio buffers for the 4 nodes
    let num_samples = 512;
    let mut ch0 = vec![0.0f32; num_samples];
    let mut ch1 = vec![0.0f32; num_samples];
    let mut ch2 = vec![0.0f32; num_samples];
    let mut ch3 = vec![0.0f32; num_samples];

    // Target at 45 deg azimuth, 180 Hz
    for i in 0..num_samples {
        let t = (i as f32) / 16000.0;
        let tone = (2.0 * PI * 180.0 * t).sin() * 0.2;
        ch0[i] = tone;
        ch1[i] = tone * 0.98;
        ch2[i] = tone * 1.02;
        ch3[i] = tone * 0.95;
    }

    let slices = [&ch0[..], &ch1[..], &ch2[..], &ch3[..]];

    // Run benchmark
    let iterations = 25;
    let start = Instant::now();
    let mut latest_rep = None;

    for _ in 0..iterations {
        let rep = engine.process_swarm_mesh_frame(&slices);
        latest_rep = rep;
    }
    let elapsed = start.elapsed();

    let total_samples = (iterations * num_samples) as f64;
    let samples_per_sec = total_samples / elapsed.as_secs_f64();
    println!(
        "Swarm Mesh Beamforming Throughput: {:.0} samples/sec ({:.1}x Real-Time at 16 kHz)",
        samples_per_sec,
        samples_per_sec / 16000.0
    );

    assert!(
        samples_per_sec > 200_000.0,
        "Throughput must exceed 200,000 samples/sec, got: {samples_per_sec:.0}"
    );

    let report = latest_rep.expect("Target report must be generated");
    assert!(engine.latest_swarm_target_report().is_some());

    // MAVLink v2 Telemetry Serialization:
    let packets = SyntheticApertureBeamformer::to_mavlink_packets(&report, 55555);
    assert_eq!(packets.len(), 4);
    assert_eq!(packets[0].name_as_str(), "SWARM_AZ");
    assert_eq!(packets[1].name_as_str(), "SWARM_EL");
    assert_eq!(packets[2].name_as_str(), "SWARM_RNG");
    assert_eq!(packets[3].name_as_str(), "SWARM_SNR");
    assert_eq!(packets[0].time_boot_ms, 55555);

    // Reset verification:
    engine.reset();
    assert!(engine.latest_swarm_target_report().is_none());
}
