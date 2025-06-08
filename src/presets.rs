//! Presets for the simulation

use std::f32::consts::PI;
use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

use crate::settings::Settings;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    user_presets_dir: PathBuf,
}

impl PresetManager {
    pub fn new() -> Self {
        let user_presets_dir = get_user_presets_dir();
        let manager = Self { 
            presets: vec![], 
            user_presets_dir,
        };
        
        // Create the user presets directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(&manager.user_presets_dir) {
            eprintln!("Warning: Could not create user presets directory: {}", e);
        }
        
        manager
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

    /// Save a preset to a TOML file in the user's Documents folder
    pub fn save_user_preset(&self, name: &str, settings: &Settings) -> Result<(), Box<dyn std::error::Error>> {
        let preset = Preset {
            name: name.to_string(),
            settings: settings.clone(),
        };
        
        let toml_content = toml::to_string_pretty(&preset)?;
        let file_path = self.user_presets_dir.join(format!("{}.toml", sanitize_filename(name)));
        fs::write(file_path, toml_content)?;
        
        Ok(())
    }

    /// Load user presets from TOML files in the user's Documents folder
    pub fn load_user_presets(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.user_presets_dir.exists() {
            return Ok(());
        }

        let entries = fs::read_dir(&self.user_presets_dir)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                match self.load_preset_from_file(&path) {
                    Ok(preset) => {
                        // Check if this preset name already exists (avoid duplicates)
                        if !self.presets.iter().any(|p| p.name == preset.name) {
                            self.presets.push(preset);
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: Could not load preset from {:?}: {}", path, e);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Load a single preset from a TOML file
    fn load_preset_from_file(&self, path: &PathBuf) -> Result<Preset, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let preset: Preset = toml::from_str(&content)?;
        Ok(preset)
    }

    /// Delete a user preset file and remove it from memory
    pub fn delete_user_preset(&mut self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let sanitized_name = sanitize_filename(name);
        let file_path = self.user_presets_dir.join(format!("{}.toml", sanitized_name));
        
        // Remove from file system
        if file_path.exists() {
            fs::remove_file(&file_path)?;
        }
        
        // Also remove from memory immediately
        self.presets.retain(|p| p.name != name);
        
        Ok(())
    }

    /// Get list of user preset files (without built-in presets)
    pub fn get_user_preset_names(&self) -> Vec<String> {
        let built_in_names = get_built_in_preset_names();
        self.presets.iter()
            .filter(|p| !built_in_names.contains(&p.name))
            .map(|p| p.name.clone())
            .collect()
    }
}

impl Default for PresetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the user's Documents folder path and create the slime-mold presets subdirectory path
fn get_user_presets_dir() -> PathBuf {
    let home_dir = std::env::home_dir()
        .unwrap_or_else(|| PathBuf::from("."));
    
    home_dir.join("Documents").join("slime-mold-presets")
}

/// Sanitize filename to be safe for filesystem
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
            _ => c,
        })
        .collect()
}

/// Get list of built-in preset names
fn get_built_in_preset_names() -> Vec<String> {
    vec![
        "Default".to_string(),
        "Gloop Loops".to_string(),
        "Firecracker Trees".to_string(),
        "Threads".to_string(),
        "Cells".to_string(),
        "Snake".to_string(),
        "Mesh".to_string(),
    ]
}

pub fn init_preset_manager() -> PresetManager {
    let mut preset_manager = PresetManager::new();
    
    // Add built-in presets
    preset_manager.add_preset(Preset::new("Default".to_string(), Settings::default()));
    preset_manager.add_preset(Preset::new(
        "Gloop Loops".to_string(),
        Settings {
            agent_jitter: 0.1,
            agent_turn_speed: 0.43,
            agent_speed_max: 300.0,
            agent_sensor_angle: 0.7,
            agent_sensor_distance: 5.0,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Firecracker Trees".to_string(),
        Settings {
            agent_jitter: 0.1,
            agent_turn_speed: 0.93,
            agent_speed_min: 200.0,
            agent_speed_max: 300.0,
            agent_sensor_angle: 0.3,
            agent_sensor_distance: 20.0,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Threads".to_string(),
        Settings {
            agent_jitter: 0.0,
            agent_turn_speed: 0.02,
            agent_sensor_angle: 0.3,
            agent_speed_max: 150.0,
            agent_sensor_distance: 20.0,
            pheromone_decay_factor: 2.0,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Cells".to_string(),
        Settings {
            agent_jitter: 0.6,
            agent_turn_speed: 3.27,
            agent_sensor_angle: PI,
            agent_sensor_distance: 195.0,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Snake".to_string(),
        Settings {
            agent_turn_speed: 0.37,
            agent_sensor_angle: 1.34,
            agent_sensor_distance: 225.0,
            ..Settings::default()
        },
    ));
    preset_manager.add_preset(Preset::new(
        "Mesh".to_string(),
        Settings {
            agent_jitter: 3.0,
            agent_turn_speed: 6.0,
            agent_sensor_angle: 1.57,
            agent_sensor_distance: 225.0,
            pheromone_decay_factor: 10.0,
            ..Settings::default()
        },
    ));
    
    // Load user presets from TOML files
    if let Err(e) = preset_manager.load_user_presets() {
        eprintln!("Warning: Could not load user presets: {}", e);
    }
    
    preset_manager
}
