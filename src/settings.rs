use std::ops::Range;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub agent_count: usize,
    pub agent_jitter: f32,
    #[serde(serialize_with = "serialize_range", deserialize_with = "deserialize_range")]
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
    // Gradient settings
    pub gradient_enabled: bool,
    pub gradient_type: GradientType,
    pub gradient_strength: f32,
    pub gradient_center_x: f32,
    pub gradient_center_y: f32,
    pub gradient_size: f32,
    pub gradient_angle: f32,
}

// Custom serialization for Range<f32>
fn serialize_range<S>(range: &Range<f32>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    (range.start, range.end).serialize(serializer)
}

fn deserialize_range<'de, D>(deserializer: D) -> Result<Range<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let (start, end) = <(f32, f32)>::deserialize(deserializer)?;
    Ok(start..end)
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GradientType {
    None,
    Linear,
    Radial,
    Ellipse,
    Spiral,
    Checkerboard,
}

impl GradientType {
    pub fn as_str(&self) -> &'static str {
        match self {
            GradientType::None => "None",
            GradientType::Linear => "Linear",
            GradientType::Radial => "Radial",
            GradientType::Ellipse => "Ellipse",
            GradientType::Spiral => "Spiral",
            GradientType::Checkerboard => "Checkerboard",
        }
    }

    pub fn all() -> &'static [GradientType] {
        &[
            GradientType::None,
            GradientType::Linear,
            GradientType::Radial,
            GradientType::Ellipse,
            GradientType::Spiral,
            GradientType::Checkerboard,
        ]
    }
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
            // Gradient defaults
            gradient_enabled: false,
            gradient_type: GradientType::None,
            gradient_strength: 0.5,
            gradient_center_x: 0.5,
            gradient_center_y: 0.5,
            gradient_size: 0.3,
            gradient_angle: 0.0,
        }
    }
}

impl Settings {
    pub fn set_agent_speed_min(&mut self, speed: f32) {
        self.agent_speed_min = speed.min(self.agent_speed_max);
    }
}
