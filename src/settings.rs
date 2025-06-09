use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// The amount of jitter to add to the agent's starting heading.
    ///
    /// Defaults to 0.04.
    pub agent_jitter: f32,
    /// The range of possible starting headings for the agent.
    ///
    /// Defaults to 0.0..360.0.
    #[serde(
        serialize_with = "serialize_range",
        deserialize_with = "deserialize_range"
    )]
    pub agent_possible_starting_headings: Range<f32>,
    /// The angle of the agent's sensor.
    ///
    /// Defaults to 0.3 radians.
    pub agent_sensor_angle: f32,
    /// The distance of the agent's sensor.
    ///
    /// Defaults to 20.0.
    pub agent_sensor_distance: f32,
    /// The maximum speed of the agent.
    ///
    /// Defaults to 60.0.
    pub agent_speed_max: f32,
    /// The minimum speed of the agent.
    ///
    /// Defaults to 30.0.
    pub agent_speed_min: f32,
    /// The speed at which the agent turns.
    ///
    /// Defaults to 0.43 rad/s.
    pub agent_turn_speed: f32,
    /// The decay factor of the pheromone.
    ///
    /// Defaults to 1.0.
    pub pheromone_decay_factor: f32,
    /// The amount of pheromone deposited by the agent.
    ///
    /// Defaults to 1.0.
    pub pheromone_deposition_amount: f32,
    /// The rate at which pheromone diffuses.
    ///
    /// Defaults to 1.0.
    pub pheromone_diffusion_rate: f32,
    /// Whether the window is fullscreen.
    ///
    /// Defaults to false.
    pub window_fullscreen: bool,
    /// The height of the window.
    ///
    /// Defaults to 900.
    pub window_height: u32,
    /// The width of the window.
    ///
    /// Defaults to 1600.
    pub window_width: u32,
    /// Whether the gradient is enabled.
    ///
    /// Defaults to false.
    pub gradient_enabled: bool,
    /// The type of gradient.
    ///
    /// Defaults to GradientType::None.
    pub gradient_type: GradientType,
    /// The strength of the gradient.
    ///
    /// Defaults to 0.5.
    pub gradient_strength: f32,
    /// The x-coordinate of the center of the gradient.
    ///
    /// Defaults to 0.5.
    pub gradient_center_x: f32,
    /// The y-coordinate of the center of the gradient.
    ///
    /// Defaults to 0.5.
    pub gradient_center_y: f32,
    /// The size of the gradient.
    ///
    /// Defaults to 0.3.
    pub gradient_size: f32,
    /// The angle of the gradient.
    ///
    /// Defaults to 0.0.
    pub gradient_angle: f32,
}

impl PartialEq for Settings {
    fn eq(&self, other: &Self) -> bool {
        const EPSILON: f32 = 1e-6; // Small epsilon for floating-point comparisons

        (self.agent_jitter - other.agent_jitter).abs() < EPSILON
            && self.agent_possible_starting_headings == other.agent_possible_starting_headings
            && (self.agent_sensor_angle - other.agent_sensor_angle).abs() < EPSILON
            && (self.agent_sensor_distance - other.agent_sensor_distance).abs() < EPSILON
            && (self.agent_speed_max - other.agent_speed_max).abs() < EPSILON
            && (self.agent_speed_min - other.agent_speed_min).abs() < EPSILON
            && (self.agent_turn_speed - other.agent_turn_speed).abs() < EPSILON
            && (self.pheromone_decay_factor - other.pheromone_decay_factor).abs() < EPSILON
            && (self.pheromone_deposition_amount - other.pheromone_deposition_amount).abs()
                < EPSILON
            && (self.pheromone_diffusion_rate - other.pheromone_diffusion_rate).abs() < EPSILON
            && self.window_fullscreen == other.window_fullscreen
            && self.window_height == other.window_height
            && self.window_width == other.window_width
            && self.gradient_enabled == other.gradient_enabled
            && self.gradient_type == other.gradient_type
            && (self.gradient_strength - other.gradient_strength).abs() < EPSILON
            && (self.gradient_center_x - other.gradient_center_x).abs() < EPSILON
            && (self.gradient_center_y - other.gradient_center_y).abs() < EPSILON
            && (self.gradient_size - other.gradient_size).abs() < EPSILON
            && (self.gradient_angle - other.gradient_angle).abs() < EPSILON
    }
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
