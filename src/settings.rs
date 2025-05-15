use crate::errors::SlimeError;
use log::info;
use serde::Deserialize;
use std::ops::Range;

// General settings
pub const WIDTH: u32 = 1600;
pub const HEIGHT: u32 = 900;
pub const IS_FULLSCREEN: bool = false;

// Agent settings
pub const AGENT_COUNT: usize = 10000;
pub const AGENT_COUNT_MAXIMUM: usize = 100000;
pub const AGENT_JITTER: f32 = 10.0;
pub const AGENT_SPEED_MIN: f32 = 0.5 + 20.0;
pub const AGENT_SPEED_MAX: f32 = 1.2 + 20.0;
pub const AGENT_TURN_SPEED: f32 = 20.0;
pub const AGENT_POSSIBLE_STARTING_HEADINGS: std::ops::Range<f32> = 0.0..360.0;
pub const DEPOSITION_AMOUNT: f32 = 1.0;

// Pheromone settings
/// Represents the rate at which pheromone signals disappear. A typical decay factor is 1/100 the rate of deposition
pub const DECAY_FACTOR: f32 = 0.01;
pub const DEFAULT_SETTINGS_FILE: &str = "simulation_settings.toml";

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub agent_count_maximum: usize,
    pub agent_count: usize,
    pub agent_deposition_amount: f32,
    pub agent_jitter: f32,
    pub agent_possible_starting_headings: Range<f32>,
    pub agent_speed_max: f32,
    pub agent_speed_min: f32,
    pub agent_turn_speed: f32,
    pub pheromone_decay_factor: f32,
    pub pheromone_enable_dynamic_gradient: bool,
    pub window_fullscreen: bool,
    pub window_height: u32,
    pub window_width: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            agent_count_maximum: AGENT_COUNT_MAXIMUM,
            agent_count: AGENT_COUNT,
            agent_deposition_amount: DEPOSITION_AMOUNT,
            agent_jitter: AGENT_JITTER,
            agent_possible_starting_headings: AGENT_POSSIBLE_STARTING_HEADINGS,
            agent_speed_max: AGENT_SPEED_MAX,
            agent_speed_min: AGENT_SPEED_MIN,
            agent_turn_speed: AGENT_TURN_SPEED,
            pheromone_decay_factor: DECAY_FACTOR,
            pheromone_enable_dynamic_gradient: true,
            window_fullscreen: IS_FULLSCREEN,
            window_height: HEIGHT,
            window_width: WIDTH,
        }
    }
}

impl Settings {
    pub fn load_from_file(settings_file_name: &str) -> Result<Self, SlimeError> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(settings_file_name))
            .build()
            .map_err(SlimeError::from)?;
        let settings = settings.try_deserialize().map_err(SlimeError::from)?;

        info!(
            "successfully loaded settings from '{}'",
            &settings_file_name
        );

        Ok(settings)
    }

    pub fn did_agent_settings_change(&self, other: &Self) -> bool {
        self.agent_count != other.agent_count
            || self.agent_count_maximum != other.agent_count_maximum
            || self.agent_jitter != other.agent_jitter
            || self.agent_speed_min != other.agent_speed_min
            || self.agent_speed_max != other.agent_speed_max
            || self.agent_turn_speed != other.agent_turn_speed
            || self.agent_possible_starting_headings != other.agent_possible_starting_headings
            || self.agent_deposition_amount != other.agent_deposition_amount
    }

    pub fn did_pheromone_settings_change(&self, other: &Self) -> bool {
        self.pheromone_decay_factor != other.pheromone_decay_factor
            || self.pheromone_enable_dynamic_gradient != other.pheromone_enable_dynamic_gradient
    }
}
