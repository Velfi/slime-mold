//! Presets for the simulation

use crate::settings::Settings;

pub struct Preset {
    pub name: String,
    pub settings: Settings,
}

impl Preset {
    pub fn new(name: String, settings: Settings) -> Self {
        Self { name, settings }
    }
}

pub struct PresetManager {
    presets: Vec<Preset>,
}

impl PresetManager {
    pub fn new() -> Self {
        Self { presets: vec![] }
    }

    pub fn add_preset(&mut self, preset: Preset) {
        self.presets.push(preset);
    }

    pub fn get_preset(&self, name: &str) -> Option<&Preset> {
        self.presets.iter().find(|p| p.name == name)
    }

    pub fn get_preset_names(&self) -> Vec<String> {
        self.presets.iter().map(|p| p.name.clone()).collect()
    }
}

impl Default for PresetManager {
    fn default() -> Self {
        Self::new()
    }
}

pub fn init_preset_manager() -> PresetManager {
    let mut preset_manager = PresetManager::new();
    preset_manager.add_preset(Preset::new("Default".to_string(), Settings::default()));
    preset_manager.add_preset(Preset::new(
        "Firecracker Trees".to_string(),
        Settings {
            agent_jitter: 0.1,
            agent_speed_min: 60.0,
            agent_speed_max: 60.0,
            agent_turn_speed: 0.8,
            agent_sensor_angle: 0.3,
            agent_sensor_distance: 20.0,
            pheromone_deposition_amount: 1.0,
            pheromone_decay_factor: 10.0,
            pheromone_diffusion_rate: 1.0,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Sponge".to_string(),
        Settings {
            agent_jitter: 0.0,
            agent_speed_min: 20.0,
            agent_speed_max: 30.0,
            agent_turn_speed: 2.0,
            agent_sensor_angle: 0.3,
            agent_sensor_distance: 20.0,
            pheromone_deposition_amount: 1.0,
            pheromone_decay_factor: 1.0,
            pheromone_diffusion_rate: 1.0,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Spiky".to_string(),
        Settings {
            agent_jitter: 0.0,
            agent_speed_min: 70.0,
            agent_speed_max: 80.0,
            agent_turn_speed: 1.0,
            agent_sensor_angle: 0.3,
            agent_sensor_distance: 20.0,
            pheromone_deposition_amount: 1.0,
            pheromone_decay_factor: 10.0,
            pheromone_diffusion_rate: 0.1,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Curls".to_string(),
        Settings {
            agent_jitter: 5.0,
            agent_speed_min: 70.0,
            agent_speed_max: 80.0,
            agent_turn_speed: 0.05,
            agent_sensor_angle: 0.3,
            agent_sensor_distance: 20.0,
            pheromone_deposition_amount: 1.0,
            pheromone_decay_factor: 10.0,
            pheromone_diffusion_rate: 0.1,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Waves".to_string(),
        Settings {
            agent_jitter: 1.0,
            agent_speed_min: 30.0,
            agent_speed_max: 50.0,
            agent_turn_speed: 6.2,
            agent_sensor_angle: 0.3,
            agent_sensor_distance: 20.0,
            pheromone_deposition_amount: 1.0,
            pheromone_decay_factor: 10.0,
            pheromone_diffusion_rate: 0.1,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Snake".to_string(),
        Settings {
            agent_jitter: 3.0,
            agent_speed_min: 100.0,
            agent_speed_max: 120.0,
            agent_turn_speed: 6.5,
            agent_sensor_angle: 1.57,
            agent_sensor_distance: 225.0,
            pheromone_deposition_amount: 1.0,
            pheromone_decay_factor: 10.0,
            pheromone_diffusion_rate: 1.0,
            ..Settings::default()
        },
    ));
    preset_manager
}
