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
pub const AGENT_JITTER: f64 = 10.0;
pub const AGENT_SPEED_MIN: f64 = 0.5 + 20.0;
pub const AGENT_SPEED_MAX: f64 = 1.2 + 20.0;
pub const AGENT_TURN_SPEED: f64 = 20.0;
pub const AGENT_POSSIBLE_STARTING_HEADINGS: std::ops::Range<f64> = 0.0..360.0;
pub const DEPOSITION_AMOUNT: u8 = u8::MAX;

// Pheromone settings
/// Represents the rate at which pheromone signals disappear. A typical decay factor is 1/100 the rate of deposition
pub const DECAY_FACTOR: u8 = u8::MIN;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    window_width: u32,
    window_height: u32,
    window_fullscreen: bool,
    agent_count: usize,
    agent_count_maximum: usize,
    agent_jitter: f64,
    agent_speed_min: f64,
    agent_speed_max: f64,
    agent_turn_speed: f64,
    agent_possible_starting_headings: Range<f64>,
    agent_deposition_amount: u8,
    pheromone_decay_factor: u8,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            window_width: WIDTH,
            window_height: HEIGHT,
            window_fullscreen: IS_FULLSCREEN,
            agent_count: AGENT_COUNT,
            agent_count_maximum: AGENT_COUNT_MAXIMUM,
            agent_jitter: AGENT_JITTER,
            agent_speed_min: AGENT_SPEED_MIN,
            agent_speed_max: AGENT_SPEED_MAX,
            agent_turn_speed: AGENT_TURN_SPEED,
            agent_possible_starting_headings: AGENT_POSSIBLE_STARTING_HEADINGS,
            agent_deposition_amount: DEPOSITION_AMOUNT,
            pheromone_decay_factor: DECAY_FACTOR,
        }
    }
}

impl Settings {
    pub fn load_from_file(settings_file_name: &str) -> Result<Self, SlimeError> {
        let mut settings = config::Config::default();
        settings.merge(config::File::with_name(settings_file_name))?;
        let settings = settings.try_into();

        info!(
            "successfully loaded settings from '{}'",
            &settings_file_name
        );

        settings.map_err(SlimeError::from)
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
    }

    pub fn window_width(&self) -> u32 {
        self.window_width
    }

    pub fn window_height(&self) -> u32 {
        self.window_height
    }

    pub fn window_fullscreen(&self) -> bool {
        self.window_fullscreen
    }

    pub fn agent_count(&self) -> usize {
        self.agent_count
    }

    pub fn agent_count_maximum(&self) -> usize {
        self.agent_count_maximum
    }

    pub fn agent_jitter(&self) -> f64 {
        self.agent_jitter
    }

    pub fn agent_speed_min(&self) -> f64 {
        self.agent_speed_min
    }

    pub fn agent_speed_max(&self) -> f64 {
        self.agent_speed_max
    }

    pub fn agent_turn_speed(&self) -> f64 {
        self.agent_turn_speed
    }

    pub fn agent_possible_starting_headings(&self) -> &Range<f64> {
        &self.agent_possible_starting_headings
    }

    pub fn agent_deposition_amount(&self) -> u8 {
        self.agent_deposition_amount
    }

    pub fn pheromone_decay_factor(&self) -> u8 {
        self.pheromone_decay_factor
    }
}
