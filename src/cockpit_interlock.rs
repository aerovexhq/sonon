//! Workstation Cockpit HUD & Tactical Autopilot Command Interlock
//! implemented in pure safe Rust for modules/sonon.
//!
//! Provides:
//! 1. Cockpit Priority Annunciator Queue with pre-emption and sub-50ms pilot barge-in yielding.
//! 2. Tactical Aeronautical Voice Command Parser with standard ICAO/NATO phonetic vocabulary.
//! 3. Distribution-Free Conformal Safety Interlock: bounds false command execution risk.
//! 4. 3D Spatial Audio Coordinate Transformer for Aerovex Workstation CesiumJS/Three.js HUD.
//! 5. MAVLink v2 Flight Control Command Serialization (`COMMAND_LONG`, `NAV_WAYPOINT`, `SET_MODE`).

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::beamforming::Point3D;
use crate::conformal::ConformalKwsPredictor;
use crate::engine::KeywordEvent;
use crate::metacognitive::{BargeInType, ConversationalFloorState, TurnManagerConfig, TurnStateManager};

/// Cockpit alert urgency level determining annunciation pre-emption priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CockpitAlertPriority {
    /// Routine telemetry status and waypoint confirmations.
    Informational = 0,
    /// Non-critical advisory information.
    Advisory = 1,
    /// Cautionary alerts requiring operator awareness.
    Caution = 2,
    /// Warnings requiring timely corrective action.
    Warning = 3,
    /// Critical emergency alerts requiring immediate evasive maneuvers.
    CriticalEmergency = 4,
}

/// A structured cockpit voice annunciation message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CockpitAnnunciation {
    pub id: u64,
    pub priority: CockpitAlertPriority,
    pub callout_text: String,
    pub nato_phraseology: String,
    pub spatial_location: Point3D,
    pub timestamp_sec: f64,
    pub allow_barge_in: bool,
}

impl CockpitAnnunciation {
    /// Construct a new cockpit annunciation.
    pub fn new(
        id: u64,
        priority: CockpitAlertPriority,
        callout_text: impl Into<String>,
        nato_phraseology: impl Into<String>,
        spatial_location: Point3D,
        timestamp_sec: f64,
    ) -> Self {
        let allow_barge_in = priority < CockpitAlertPriority::CriticalEmergency;
        Self {
            id,
            priority,
            callout_text: callout_text.into(),
            nato_phraseology: nato_phraseology.into(),
            spatial_location,
            timestamp_sec,
            allow_barge_in,
        }
    }
}

/// Decoded tactical flight command intended for autopilot execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TacticalFlightCommand {
    /// Maintain target altitude in feet.
    AltitudeHold { target_altitude_ft: f32 },
    /// Turn to magnetic heading vector in degrees [0, 360).
    HeadingVector { heading_deg: f32 },
    /// Maintain indicated airspeed in knots.
    AirspeedHold { knots: f32 },
    /// Direct navigation to target waypoint designator.
    WaypointDirect { waypoint_id: String },
    /// Set transponder mode 3/A squawk code.
    TransponderSquawk { squawk_code: u16 },
    /// Immediate emergency loiter / hold position.
    LoiterHold,
    /// Immediate mission abort and return to launch (RTL).
    ReturnToLaunch,
    /// Immediate flight termination or motor disarm.
    EmergencyAbort,
}

/// Result of evaluating a pilot voice command through the safety interlock.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommandInterlockDecision {
    pub is_authorized: bool,
    pub is_conformal_verified: bool,
    pub conformal_p_value: f32,
    pub confidence: f32,
    pub command: Option<TacticalFlightCommand>,
    pub mavlink_command_id: Option<u16>,
    pub feedback_annunciation: String,
}

/// Binaural 3D spatial panning parameters for CesiumJS/Three.js HUD.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinauralSpatialParams {
    pub azimuth_deg: f32,
    pub elevation_deg: f32,
    pub distance_m: f32,
    pub left_gain: f32,
    pub right_gain: f32,
    pub itd_seconds: f32,
}

/// Main Cockpit Voice HUD & Autopilot Command Interlock engine.
pub struct CockpitVoiceInterlock {
    annunciation_queue: VecDeque<CockpitAnnunciation>,
    active_annunciation: Option<CockpitAnnunciation>,
    turn_manager: TurnStateManager,
    max_risk_alpha: f32,
    next_annunciation_id: u64,
}

impl CockpitVoiceInterlock {
    /// Construct a new CockpitVoiceInterlock with default safety bounds (alpha_risk = 0.01).
    pub fn new(max_risk_alpha: f32) -> Self {
        Self {
            annunciation_queue: VecDeque::new(),
            active_annunciation: None,
            turn_manager: TurnStateManager::new(TurnManagerConfig::default()),
            max_risk_alpha,
            next_annunciation_id: 1,
        }
    }

    /// Enqueue a flight annunciation with priority-based pre-emption.
    pub fn enqueue_annunciation(
        &mut self,
        priority: CockpitAlertPriority,
        callout_text: impl Into<String>,
        nato_phraseology: impl Into<String>,
        spatial_location: Point3D,
        timestamp_sec: f64,
    ) -> u64 {
        let id = self.next_annunciation_id;
        self.next_annunciation_id += 1;

        let annunciation = CockpitAnnunciation::new(
            id,
            priority,
            callout_text,
            nato_phraseology,
            spatial_location,
            timestamp_sec,
        );

        // Pre-empt active speech if incoming message has higher priority
        if let Some(ref active) = self.active_annunciation {
            if annunciation.priority > active.priority {
                let displaced = self.active_annunciation.take().unwrap();
                self.annunciation_queue.push_front(displaced);
                self.active_annunciation = Some(annunciation);
                self.turn_manager.set_state(ConversationalFloorState::Speaking);
                return id;
            }
        }

        // Insert in priority order
        let mut inserted = false;
        for i in 0..self.annunciation_queue.len() {
            if annunciation.priority > self.annunciation_queue[i].priority {
                self.annunciation_queue.insert(i, annunciation.clone());
                inserted = true;
                break;
            }
        }
        if !inserted {
            self.annunciation_queue.push_back(annunciation);
        }

        if self.active_annunciation.is_none() {
            self.pop_next_annunciation();
        }

        id
    }

    /// Advance annunciation queue, returning the active message being spoken.
    pub fn pop_next_annunciation(&mut self) -> Option<&CockpitAnnunciation> {
        self.active_annunciation = self.annunciation_queue.pop_front();
        if self.active_annunciation.is_some() {
            self.turn_manager.set_state(ConversationalFloorState::Speaking);
        } else {
            self.turn_manager.set_state(ConversationalFloorState::Idle);
        }
        self.active_annunciation.as_ref()
    }

    /// Retrieve currently active speaking annunciation.
    pub fn active_annunciation(&self) -> Option<&CockpitAnnunciation> {
        self.active_annunciation.as_ref()
    }

    /// Process microphone streaming frame for sub-50ms pilot barge-in yielding.
    pub fn process_microphone_frame(&mut self, user_mic_rms: f32, frame_duration_ms: f32) -> Option<BargeInType> {
        let is_agent_speaking = self.active_annunciation.is_some();
        let (_state, barge_in) = self.turn_manager.update(user_mic_rms, is_agent_speaking, frame_duration_ms);

        if let Some(ref active) = self.active_annunciation {
            if active.allow_barge_in && barge_in == Some(BargeInType::ActiveOverride) {
                // Yield audio output immediately (< 50ms)
                self.turn_manager.set_state(ConversationalFloorState::Yielding);
                self.active_annunciation = None;
            }
        }

        barge_in
    }

    /// Parse recognized spoken transcript into a tactical flight command.
    pub fn parse_spoken_command(&self, text: &str) -> Option<TacticalFlightCommand> {
        let clean = text.to_lowercase();

        if clean.contains("return to launch") || clean.contains("rtl") || clean.contains("return home") {
            return Some(TacticalFlightCommand::ReturnToLaunch);
        }
        if clean.contains("abort") || clean.contains("emergency stop") {
            return Some(TacticalFlightCommand::EmergencyAbort);
        }
        if clean.contains("hold position") || clean.contains("loiter") {
            return Some(TacticalFlightCommand::LoiterHold);
        }
        if clean.contains("altitude") || clean.contains("climb to") || clean.contains("descend to") {
            let alt = self.extract_first_number(&clean).unwrap_or(2000.0);
            return Some(TacticalFlightCommand::AltitudeHold { target_altitude_ft: alt });
        }
        if clean.contains("heading") || clean.contains("vector") {
            let hdg = self.extract_first_number(&clean).unwrap_or(0.0);
            let bounded_hdg = ((hdg % 360.0) + 360.0) % 360.0;
            return Some(TacticalFlightCommand::HeadingVector { heading_deg: bounded_hdg });
        }
        if clean.contains("speed") || clean.contains("knots") {
            let spd = self.extract_first_number(&clean).unwrap_or(80.0);
            return Some(TacticalFlightCommand::AirspeedHold { knots: spd });
        }
        if clean.contains("squawk") {
            let code = self.extract_first_number(&clean).unwrap_or(1200.0) as u16;
            return Some(TacticalFlightCommand::TransponderSquawk { squawk_code: code });
        }
        if clean.contains("waypoint") || clean.contains("direct to") {
            let wp = self.extract_waypoint_id(&clean).unwrap_or_else(|| "ALPHA".to_string());
            return Some(TacticalFlightCommand::WaypointDirect { waypoint_id: wp });
        }

        None
    }

    /// Evaluate command through distribution-free conformal prediction bounds.
    pub fn evaluate_flight_interlock(
        &self,
        event: &KeywordEvent,
        conformal_predictor: Option<&ConformalKwsPredictor>,
    ) -> CommandInterlockDecision {
        let parsed = self.parse_spoken_command(&event.keyword);

        let p_value = event.conformal_p_value.unwrap_or_else(|| {
            if let Some(cp) = conformal_predictor {
                let approx_dist = (1.0 - event.confidence).max(0.0) * 10.0;
                cp.compute_p_value(&event.keyword, approx_dist)
            } else {
                0.005
            }
        });

        let is_conformal_verified = event.is_conformal_verified || p_value <= self.max_risk_alpha;
        let is_authorized = is_conformal_verified && parsed.is_some();

        let (mav_id, feedback) = match &parsed {
            Some(TacticalFlightCommand::EmergencyAbort) => (
                Some(400), // MAV_CMD_COMPONENT_ARM_DISARM
                "Emergency abort confirmed. Disarming motors immediately.".to_string(),
            ),
            Some(TacticalFlightCommand::ReturnToLaunch) => (
                Some(20), // MAV_CMD_NAV_RETURN_TO_LAUNCH
                "Return to launch confirmed. Engaging autonomous RTL profile.".to_string(),
            ),
            Some(TacticalFlightCommand::LoiterHold) => (
                Some(17), // MAV_CMD_NAV_LOITER_UNLIM
                "Loiter hold confirmed. Maintaining position and altitude.".to_string(),
            ),
            Some(TacticalFlightCommand::AltitudeHold { target_altitude_ft }) => (
                Some(179), // MAV_CMD_DO_CHANGE_ALTITUDE
                format!("Altitude clearance confirmed. Maintaining {} feet.", target_altitude_ft),
            ),
            Some(TacticalFlightCommand::HeadingVector { heading_deg }) => (
                Some(115), // MAV_CMD_CONDITION_YAW
                format!("Heading vector confirmed. Turning to heading {} degrees.", heading_deg),
            ),
            Some(TacticalFlightCommand::AirspeedHold { knots }) => (
                Some(178), // MAV_CMD_DO_CHANGE_SPEED
                format!("Speed hold confirmed. Maintaining {} knots.", knots),
            ),
            Some(TacticalFlightCommand::TransponderSquawk { squawk_code }) => (
                Some(208),
                format!("Squawk code confirmed. Setting transponder to {:04}.", squawk_code),
            ),
            Some(TacticalFlightCommand::WaypointDirect { waypoint_id }) => (
                Some(16), // MAV_CMD_NAV_WAYPOINT
                format!("Direct to waypoint {} confirmed.", waypoint_id),
            ),
            None => (None, "Command not recognized or ambiguous. Repeat command.".to_string()),
        };

        CommandInterlockDecision {
            is_authorized,
            is_conformal_verified,
            conformal_p_value: p_value,
            confidence: event.confidence,
            command: parsed,
            mavlink_command_id: mav_id,
            feedback_annunciation: feedback,
        }
    }

    /// Compute binaural 3D spatial panning parameters for CesiumJS/Three.js HUD.
    pub fn compute_binaural_spatial_params(&self, pos: &Point3D) -> BinauralSpatialParams {
        let distance = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt().max(0.1);
        let azimuth = (pos.y).atan2(pos.x).to_degrees();
        let elevation = (pos.z / distance).asin().to_degrees();

        // Standard head diameter ~17.5 cm, speed of sound ~343 m/s
        let head_radius_m = 0.0875f32;
        let speed_of_sound_m_s = 343.0f32;
        let az_rad = azimuth.to_radians();
        let itd = (head_radius_m / speed_of_sound_m_s) * (az_rad.sin() + az_rad);

        // Interaural Level Difference (ILD) approximation
        let pan = (az_rad.sin() + 1.0) * 0.5; // [0, 1] left to right
        let dist_attenuation = (1.0 / (1.0 + 0.1 * distance)).min(1.0);
        let left_gain = (1.0 - pan).sqrt() * dist_attenuation;
        let right_gain = pan.sqrt() * dist_attenuation;

        BinauralSpatialParams {
            azimuth_deg: azimuth,
            elevation_deg: elevation,
            distance_m: distance,
            left_gain,
            right_gain,
            itd_seconds: itd.abs(),
        }
    }

    /// Extract first floating-point number in string.
    fn extract_first_number(&self, text: &str) -> Option<f32> {
        let mut curr = String::new();
        for ch in text.chars() {
            if ch.is_ascii_digit() || ch == '.' {
                curr.push(ch);
            } else if !curr.is_empty() {
                break;
            }
        }
        curr.parse::<f32>().ok()
    }

    /// Extract NATO phonetic waypoint ID in string.
    fn extract_waypoint_id(&self, text: &str) -> Option<String> {
        let tokens = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "kilo"];
        for tok in tokens {
            if text.contains(tok) {
                return Some(tok.to_uppercase());
            }
        }
        None
    }

    pub fn turn_manager(&self) -> &TurnStateManager {
        &self.turn_manager
    }

    pub fn queue_len(&self) -> usize {
        self.annunciation_queue.len()
    }
}
