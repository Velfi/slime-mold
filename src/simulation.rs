use bytemuck::{Pod, Zeroable};
use crate::settings::Settings;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SimSizeUniform {
    pub width: u32,
    pub height: u32,
    pub decay_factor: f32,
    pub agent_jitter: f32,
    pub agent_speed_min: f32,
    pub agent_speed_max: f32,
    pub agent_turn_speed: f32,
    pub agent_sensor_angle: f32,
    pub agent_sensor_distance: f32,
    pub diffusion_rate: f32,
    pub pheromone_deposition_amount: f32,
    pub _pad: [u32; 3],
}

impl SimSizeUniform {
    pub fn new(width: u32, height: u32, decay_factor: f32, settings: &Settings) -> Self {
        Self {
            width,
            height,
            decay_factor,
            agent_jitter: settings.agent_jitter,
            agent_speed_min: settings.agent_speed_min,
            agent_speed_max: settings.agent_speed_max,
            agent_turn_speed: settings.agent_turn_speed,
            agent_sensor_angle: settings.agent_sensor_angle,
            agent_sensor_distance: settings.agent_sensor_distance,
            diffusion_rate: settings.pheromone_diffusion_rate,
            pheromone_deposition_amount: settings.pheromone_deposition_amount,
            _pad: [0, 0, 0],
        }
    }
} 