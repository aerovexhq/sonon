//! Integration tests for Phase 9: Workstation Cockpit HUD & Tactical Autopilot Interlock
//! in modules/sonon.

use sonon::beamforming::Point3D;
use sonon::cockpit_interlock::{CockpitAlertPriority, CockpitVoiceInterlock, TacticalFlightCommand};
use sonon::engine::{KeywordEvent, SononEngine};
use sonon::metacognitive::BargeInType;

#[test]
fn test_cockpit_annunciator_priority_preemption() {
    let mut interlock = CockpitVoiceInterlock::new(0.01);

    // 1. Enqueue routine advisory
    let id1 = interlock.enqueue_annunciation(
        CockpitAlertPriority::Advisory,
        "Waypoint Alpha reached",
        "WAYPOINT ALPHA REACHED",
        Point3D::new(10.0, 0.0, 0.0),
        100.0,
    );
    assert_eq!(id1, 1);
    assert_eq!(
        interlock.active_annunciation().unwrap().callout_text,
        "Waypoint Alpha reached"
    );

    // 2. Enqueue caution alert - should queue behind active
    let id2 = interlock.enqueue_annunciation(
        CockpitAlertPriority::Caution,
        "Rotor temperature elevated",
        "ROTOR TEMPERATURE ELEVATED",
        Point3D::new(5.0, 2.0, 0.0),
        100.1,
    );
    assert_eq!(id2, 2);

    // 3. Enqueue CriticalEmergency - should PRE-EMPT active advisory immediately
    let id3 = interlock.enqueue_annunciation(
        CockpitAlertPriority::CriticalEmergency,
        "Terrain terrain pull up",
        "TERRAIN TERRAIN PULL UP",
        Point3D::new(0.0, 0.0, -10.0),
        100.2,
    );
    assert_eq!(id3, 3);

    // Active speech is now the emergency callout
    let active = interlock.active_annunciation().unwrap();
    assert_eq!(active.id, 3);
    assert_eq!(active.priority, CockpitAlertPriority::CriticalEmergency);
    assert_eq!(active.callout_text, "Terrain terrain pull up");
    assert!(!active.allow_barge_in);

    // 4. Pop next: after emergency finishes, highest priority in queue (Caution id2) speaks next
    let next1 = interlock.pop_next_annunciation().unwrap();
    assert_eq!(next1.id, 2);
    assert_eq!(next1.priority, CockpitAlertPriority::Caution);

    // 5. Pop next: displaced Advisory id1 speaks last
    let next2 = interlock.pop_next_annunciation().unwrap();
    assert_eq!(next2.id, 1);
    assert_eq!(next2.priority, CockpitAlertPriority::Advisory);

    // Queue is now empty
    assert!(interlock.pop_next_annunciation().is_none());
}

#[test]
fn test_tactical_flight_command_parsing() {
    let interlock = CockpitVoiceInterlock::new(0.01);

    // Altitude hold
    let cmd = interlock.parse_spoken_command("aerovex climb to 4500 feet");
    assert_eq!(
        cmd,
        Some(TacticalFlightCommand::AltitudeHold {
            target_altitude_ft: 4500.0
        })
    );

    // Heading vector
    let cmd = interlock.parse_spoken_command("aerovex turn heading 180");
    assert_eq!(
        cmd,
        Some(TacticalFlightCommand::HeadingVector {
            heading_deg: 180.0
        })
    );

    // Airspeed hold
    let cmd = interlock.parse_spoken_command("set speed 95 knots");
    assert_eq!(
        cmd,
        Some(TacticalFlightCommand::AirspeedHold { knots: 95.0 })
    );

    // Return to launch (RTL)
    let cmd = interlock.parse_spoken_command("mission abort return to launch");
    assert_eq!(cmd, Some(TacticalFlightCommand::ReturnToLaunch));

    // Emergency abort / motor cut
    let cmd = interlock.parse_spoken_command("emergency abort flight now");
    assert_eq!(cmd, Some(TacticalFlightCommand::EmergencyAbort));

    // Transponder squawk
    let cmd = interlock.parse_spoken_command("squawk 7700 emergency");
    assert_eq!(
        cmd,
        Some(TacticalFlightCommand::TransponderSquawk {
            squawk_code: 7700
        })
    );

    // Waypoint direct
    let cmd = interlock.parse_spoken_command("direct to waypoint charlie");
    assert_eq!(
        cmd,
        Some(TacticalFlightCommand::WaypointDirect {
            waypoint_id: "CHARLIE".to_string()
        })
    );
}

#[test]
fn test_conformal_safety_interlock_gating() {
    let interlock = CockpitVoiceInterlock::new(0.01); // 1% false alarm risk bound

    // Case 1: High confidence, conformal verified command
    let mut verified_event = KeywordEvent::new("climb to 3000 feet", 0.98, 120.0);
    verified_event.conformal_p_value = Some(0.003); // p <= 0.01
    verified_event.is_conformal_verified = true;

    let decision = interlock.evaluate_flight_interlock(&verified_event, None);
    assert!(decision.is_authorized);
    assert!(decision.is_conformal_verified);
    assert_eq!(decision.mavlink_command_id, Some(179)); // MAV_CMD_DO_CHANGE_ALTITUDE
    assert_eq!(
        decision.command,
        Some(TacticalFlightCommand::AltitudeHold {
            target_altitude_ft: 3000.0
        })
    );

    // Case 2: Ambiguous or unverified candidate (p-value = 0.05 > 0.01 alpha bound)
    let mut unverified_event = KeywordEvent::new("climb to 3000 feet", 0.65, 121.0);
    unverified_event.conformal_p_value = Some(0.05); // Violates 1% bound
    unverified_event.is_conformal_verified = false;

    let decision_bad = interlock.evaluate_flight_interlock(&unverified_event, None);
    assert!(!decision_bad.is_authorized);
    assert!(!decision_bad.is_conformal_verified);

    // Case 3: Emergency abort command verification
    let mut abort_event = KeywordEvent::new("abort", 0.99, 122.0);
    abort_event.conformal_p_value = Some(0.001);
    abort_event.is_conformal_verified = true;

    let decision_abort = interlock.evaluate_flight_interlock(&abort_event, None);
    assert!(decision_abort.is_authorized);
    assert_eq!(decision_abort.mavlink_command_id, Some(400)); // MAV_CMD_COMPONENT_ARM_DISARM
    assert_eq!(
        decision_abort.command,
        Some(TacticalFlightCommand::EmergencyAbort)
    );
}

#[test]
fn test_hud_3d_spatial_audio_coordinate_transform() {
    let interlock = CockpitVoiceInterlock::new(0.01);

    // Target straight ahead: Azimuth ~0 deg, balanced gains, minimal ITD
    let front = Point3D::new(10.0, 0.0, 0.0);
    let params_front = interlock.compute_binaural_spatial_params(&front);
    assert!(params_front.azimuth_deg.abs() < 1.0);
    assert!((params_front.left_gain - params_front.right_gain).abs() < 0.1);
    assert!(params_front.itd_seconds < 0.0001);

    // Target 90 deg right (positive Y in coordinates): Right gain > Left gain
    let right = Point3D::new(0.0, 10.0, 0.0);
    let params_right = interlock.compute_binaural_spatial_params(&right);
    assert!(params_right.azimuth_deg > 80.0);
    assert!(params_right.right_gain > params_right.left_gain);
    assert!(params_right.itd_seconds > 0.0002);

    // Target 90 deg left (negative Y): Left gain > Right gain
    let left = Point3D::new(0.0, -10.0, 0.0);
    let params_left = interlock.compute_binaural_spatial_params(&left);
    assert!(params_left.azimuth_deg < -80.0);
    assert!(params_left.left_gain > params_left.right_gain);
}

#[test]
fn test_sub_50ms_barge_in_yielding() {
    let mut interlock = CockpitVoiceInterlock::new(0.01);

    // 1. Enqueue advisory annunciation (allow_barge_in = true)
    interlock.enqueue_annunciation(
        CockpitAlertPriority::Advisory,
        "Transponder radar contact established",
        "TRANSPONDER RADAR CONTACT",
        Point3D::new(5.0, 0.0, 0.0),
        50.0,
    );
    assert!(interlock.active_annunciation().is_some());

    // 2. Pilot speaks with high vocal energy (0.12 RMS >= 0.08 ActiveOverride threshold)
    let barge_in = interlock.process_microphone_frame(0.12, 20.0);
    assert_eq!(barge_in, Some(BargeInType::ActiveOverride));

    // Active annunciation immediately yielded / muted
    assert!(interlock.active_annunciation().is_none());

    // 3. Test CriticalEmergency annunciation (allow_barge_in = false)
    interlock.enqueue_annunciation(
        CockpitAlertPriority::CriticalEmergency,
        "Warning pull up terrain",
        "WARNING PULL UP",
        Point3D::new(0.0, 0.0, -5.0),
        51.0,
    );
    assert!(interlock.active_annunciation().is_some());

    // Pilot speaks again - CriticalEmergency must NOT yield
    let barge_in_emerg = interlock.process_microphone_frame(0.15, 20.0);
    assert_eq!(barge_in_emerg, Some(BargeInType::ActiveOverride));
    assert!(interlock.active_annunciation().is_some());
    assert_eq!(
        interlock.active_annunciation().unwrap().priority,
        CockpitAlertPriority::CriticalEmergency
    );
}

#[test]
fn test_sonon_engine_cockpit_interlock_integration() {
    let mut engine = SononEngine::new(16000.0, 512, 160, 13);

    // Initially interlock is None
    assert!(engine.cockpit_interlock().is_none());

    // Enable cockpit interlock with 0.01 risk bound
    engine.enable_cockpit_interlock(0.01);
    assert!(engine.cockpit_interlock().is_some());

    // Enqueue annunciation via engine
    let ann_id = engine
        .enqueue_cockpit_annunciation(
            CockpitAlertPriority::Caution,
            "Low battery 20 percent remaining",
            "LOW BATTERY 20 PERCENT",
            Point3D::new(0.0, -2.0, 1.0),
            200.0,
        )
        .unwrap();
    assert_eq!(ann_id, 1);

    // Compute HUD spatial audio via engine
    let hud_audio = engine
        .compute_hud_spatial_audio(&Point3D::new(10.0, 0.0, 0.0))
        .unwrap();
    assert!(hud_audio.distance_m >= 9.9);

    // Process pilot voice command via engine
    let mut cmd_event = KeywordEvent::new("return to launch", 0.96, 201.0);
    cmd_event.is_conformal_verified = true;
    let decision = engine.process_cockpit_voice_command(&cmd_event).unwrap();
    assert!(decision.is_authorized);
    assert_eq!(decision.mavlink_command_id, Some(20)); // MAV_CMD_NAV_RETURN_TO_LAUNCH

    // Process microphone frame for barge-in via engine
    let barge_in_res = engine.process_cockpit_microphone_frame(0.10, 20.0).unwrap();
    assert_eq!(barge_in_res, Some(BargeInType::ActiveOverride));

    // Disable cockpit interlock
    engine.disable_cockpit_interlock();
    assert!(engine.cockpit_interlock().is_none());
}
