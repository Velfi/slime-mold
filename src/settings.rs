use log::info;
use serde::Deserialize;
use std::ops::Range;

// General settings
pub const WIDTH: u32 = 1600;
pub const HEIGHT: u32 = 900;
pub const IS_FULLSCREEN: bool = false;

// Agent settings
pub const AGENT_COUNT: usize = 10000;
pub const AGENT_SPEED_MIN: f32 = 30.0;
pub const AGENT_SPEED_MAX: f32 = 50.0;
pub const AGENT_TURN_SPEED: f32 = 10.0;
pub const AGENT_POSSIBLE_STARTING_HEADINGS: std::ops::Range<f32> = 0.0..360.0;
pub const DEPOSITION_AMOUNT: f32 = 1.0;
pub const AGENT_JITTER: f32 = 10.0;
pub const AGENT_SENSOR_ANGLE: f32 = 0.5;
pub const AGENT_SENSOR_DISTANCE: f32 = 9.0;

// Pheromone settings
/// Represents the rate at which pheromone signals disappear. A typical decay factor is 1/100 the rate of deposition
pub const DECAY_FACTOR: f32 = 0.01;
/// Represents how quickly pheromones diffuse to neighboring cells
pub const DIFFUSION_RATE: f32 = 0.1;
pub const DEFAULT_SETTINGS_FILE: &str = "simulation_settings.toml";

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub agent_count: usize,
    pub agent_jitter: f32,
    pub agent_possible_starting_headings: Range<f32>,
    pub agent_speed_max: f32,
    pub agent_speed_min: f32,
    pub agent_turn_speed: f32,
    pub pheromone_decay_factor: f32,
    pub pheromone_diffusion_rate: f32,
    pub pheromone_deposition_amount: f32,
    pub window_fullscreen: bool,
    pub window_height: u32,
    pub window_width: u32,
    pub agent_sensor_angle: f32,
    pub agent_sensor_distance: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            agent_count: AGENT_COUNT,
            agent_jitter: AGENT_JITTER,
            agent_possible_starting_headings: AGENT_POSSIBLE_STARTING_HEADINGS,
            agent_speed_max: AGENT_SPEED_MAX,
            agent_speed_min: AGENT_SPEED_MIN,
            agent_turn_speed: AGENT_TURN_SPEED,
            pheromone_decay_factor: DECAY_FACTOR,
            pheromone_diffusion_rate: DIFFUSION_RATE,
            pheromone_deposition_amount: DEPOSITION_AMOUNT,
            window_fullscreen: IS_FULLSCREEN,
            window_height: HEIGHT,
            window_width: WIDTH,
            agent_sensor_angle: AGENT_SENSOR_ANGLE,
            agent_sensor_distance: AGENT_SENSOR_DISTANCE,
        }
    }
}

impl Settings {
    pub fn load_from_file(settings_file_name: &str) -> anyhow::Result<Self> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(settings_file_name))
            .build()?;
        let settings = settings.try_deserialize()?;

        info!(
            "successfully loaded settings from '{}'",
            &settings_file_name
        );

        Ok(settings)
    }
}
