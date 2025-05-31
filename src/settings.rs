use std::ops::Range;

#[derive(Debug, Clone)]
pub struct Settings {
    pub agent_count: usize,
    pub agent_jitter: f32,
    pub agent_possible_starting_headings: Range<f32>,
    pub agent_sensor_angle: f32,
    pub agent_sensor_distance: f32,
    pub agent_speed_max: f32,
    pub agent_speed_min: f32,
    pub agent_turn_speed: f32,
    pub pheromone_decay_factor: f32,
    pub pheromone_deposition_amount: f32,
    pub pheromone_diffusion_rate: f32,
    pub window_fullscreen: bool,
    pub window_height: u32,
    pub window_width: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            agent_count: 1_000_000,
            agent_jitter: 0.04,
            agent_possible_starting_headings: 0.0..360.0,
            agent_sensor_angle: 0.3,
            agent_sensor_distance: 20.0,
            agent_speed_max: 60.0,
            agent_speed_min: 30.0,
            agent_turn_speed: 0.43, // ~25 degrees
            pheromone_decay_factor: 1.0,
            pheromone_deposition_amount: 1.0,
            pheromone_diffusion_rate: 1.0,
            window_fullscreen: false,
            window_height: 900,
            window_width: 1600,
        }
    }
}

impl Settings {
    pub fn set_agent_speed_min(&mut self, speed: f32) {
        self.agent_speed_min = speed.min(self.agent_speed_max);
    }
}
