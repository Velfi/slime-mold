use std::ops::Range;

use crate::{Point2, pheromones::Pheromones, rect::Rect, settings::Settings};
use log::trace;
use num::{Float, NumCast};
use rand::prelude::*;
use rand::rngs::StdRng;
use typed_builder::TypedBuilder;

pub type SensorReading = (f32, f32, f32);

#[derive(TypedBuilder)]
pub struct Agent {
    location: Point2,
    // The heading an agent is facing. (In degrees)
    #[builder(default)]
    heading: f32,
    // There are three sensors per agent: A center sensor, a left sensor, and a right sensor. The side sensors are positioned based on this angle. (In degrees)
    #[builder(default = 45.0f32)]
    sensor_angle: f32,
    // How far out a sensor is from the agent
    #[builder(default = 9.0f32)]
    sensor_distance: f32,
    // How far out a sensor is from the agent
    #[builder(default = 1.0f32)]
    move_speed: f32,
    // How quickly the agent can rotate
    #[builder(default = 20.0f32)]
    rotation_speed: f32,
    // The tendency of agents to move erratically
    #[builder(default = 0.0f32)]
    jitter: f32,
    #[builder(default = default_rng())]
    rng: StdRng,
    deposition_amount: f32,
}

#[derive(Default)]
pub struct AgentUpdate {
    pub location: Option<Point2>,
    pub heading: Option<f32>,
    pub sensor_angle: Option<f32>,
    pub sensor_distance: Option<f32>,
    pub move_speed: Option<f32>,
    pub rotation_speed: Option<f32>,
    pub jitter: Option<f32>,
    pub deposition_amount: Option<f32>,
}

impl Agent {
    pub fn new_from_settings(settings: &Settings) -> Self {
        let mut rng: StdRng = StdRng::from_os_rng();
        let deposition_amount = settings.agent_deposition_amount;
        let move_speed = rng.random_range(settings.agent_speed_min..settings.agent_speed_max);
        let location = Point2::new(
            rng.random_range(0.0..(settings.window_width as f32)),
            rng.random_range(0.0..(settings.window_height as f32)),
        );
        let heading = rng.random_range(settings.agent_possible_starting_headings.clone());

        Agent::builder()
            .location(location)
            .heading(heading)
            .move_speed(move_speed)
            .jitter(settings.agent_jitter)
            .deposition_amount(deposition_amount)
            .rotation_speed(settings.agent_turn_speed)
            .rng(StdRng::from_os_rng())
            .build()
    }

    pub fn update(&mut self, pheromones: &Pheromones, delta_t: f32, boundary_rect: &Rect<u32>) {
        let sensory_input = self.sense(pheromones, boundary_rect);
        let rotation_towards_sensory_input = self.judge_sensory_input(sensory_input);
        self.rotate(rotation_towards_sensory_input);

        move_in_direction_of_heading(
            &mut self.location,
            self.heading,
            self.move_speed,
            delta_t,
            boundary_rect,
        );
    }

    pub fn judge_sensory_input(&mut self, (l_reading, c_reading, r_reading): SensorReading) -> f32 {
        if c_reading > l_reading && c_reading > r_reading {
            // do nothing, stay facing same direction
            trace!("Agent's center value is greatest, doing nothing");
            0.0
        } else if c_reading < l_reading && c_reading < r_reading {
            // rotate randomly to the left or right
            let should_rotate_right: bool = self.rng.random();

            if should_rotate_right {
                trace!("Agent is rotating randomly to the right");
                self.rotation_speed
            } else {
                trace!("Agent is rotating randomly to the left");
                -self.rotation_speed
            }
        } else if l_reading < r_reading {
            // rotate right
            trace!("Agent is rotating right");
            self.rotation_speed
        } else if r_reading < l_reading {
            // rotate left
            trace!("Agent is rotating left");
            -self.rotation_speed
        } else {
            trace!("Agent is doing nothing (final fallthrough case)");
            0.0
        }
    }

    pub fn location(&self) -> Point2 {
        self.location
    }

    pub fn deposition_amount(&self) -> f32 {
        self.deposition_amount
    }

    pub fn set_new_random_move_speed_in_range(&mut self, move_speed_range: Range<f32>) {
        self.move_speed = self.rng.random_range(move_speed_range);
        trace!("set agent's speed to {}", self.move_speed);
    }

    pub fn sense(&self, pheromones: &Pheromones, boundary_rect: &Rect<u32>) -> SensorReading {
        let sensor_l_loc_wrapped = calculate_wrapped_sensor_location(
            self.location,
            self.heading,
            -self.sensor_angle,
            self.sensor_distance,
            boundary_rect,
        );
        let sensor_c_loc_wrapped = calculate_wrapped_sensor_location(
            self.location,
            self.heading,
            0.0, // Center sensor is straight ahead
            self.sensor_distance,
            boundary_rect,
        );
        let sensor_r_loc_wrapped = calculate_wrapped_sensor_location(
            self.location,
            self.heading,
            self.sensor_angle,
            self.sensor_distance,
            boundary_rect,
        );

        // This assumes that there is a 1:1 relationship between an agent's possible
        // movement space in a grid, and the pheromone field. What if we want to have agents moving
        // around and storing that at one level of detail and save the pheromone field at another level
        // of detail?
        // With wrapping, sensors should now always read from within the pheromone grid.
        let sensor_l_reading = pheromones.get_reading(sensor_l_loc_wrapped).unwrap_or(0) as f32;
        let sensor_c_reading = pheromones.get_reading(sensor_c_loc_wrapped).unwrap_or(0) as f32;
        let sensor_r_reading = pheromones.get_reading(sensor_r_loc_wrapped).unwrap_or(0) as f32;

        (sensor_l_reading, sensor_c_reading, sensor_r_reading)
    }

    pub fn rotate(&mut self, mut rotation_in_degrees: f32) {
        if self.jitter != 0.0 {
            let magnitude = if self.rng.random() {
                self.rng.random::<f32>()
            } else {
                self.rng.random::<f32>() * -1.0
            };
            // Randomly adjust rotation amount
            rotation_in_degrees += self.jitter * magnitude;
        }

        self.heading = rotate_by_degrees(self.heading, rotation_in_degrees);
        trace!("new heading is {}", self.heading);
    }

    pub fn apply_update(&mut self, update: &AgentUpdate) {
        if let Some(val) = update.location {
            self.location = val;
        }
        if let Some(val) = update.heading {
            self.heading = val;
        }
        if let Some(val) = update.sensor_angle {
            self.sensor_angle = val;
        }
        if let Some(val) = update.sensor_distance {
            self.sensor_distance = val;
        }
        if let Some(val) = update.move_speed {
            self.move_speed = val;
        }
        if let Some(val) = update.rotation_speed {
            self.rotation_speed = val;
        }
        if let Some(val) = update.jitter {
            self.jitter = val;
        }
        if let Some(val) = update.deposition_amount {
            self.deposition_amount = val;
        }
    }

    pub fn scale_location(
        &mut self,
        old_width: f32,
        old_height: f32,
        new_width: f32,
        new_height: f32,
    ) {
        let scale_x = new_width / old_width;
        let scale_y = new_height / old_height;
        self.location.x *= scale_x;
        self.location.y *= scale_y;
    }
}

fn rotate_by_degrees<T: Float>(n: T, rotation_in_degrees: T) -> T {
    let n_mod = NumCast::from(360.0).unwrap();
    let zero = NumCast::from(0.0).unwrap();
    let mut rotated_n = n + rotation_in_degrees;

    loop {
        if rotated_n > n_mod {
            rotated_n = rotated_n - n_mod;
        } else if rotated_n < zero {
            rotated_n = rotated_n + n_mod;
        } else {
            break;
        }
    }

    rotated_n
}

fn calculate_wrapped_sensor_location(
    agent_location: Point2,
    agent_heading: f32,
    sensor_relative_angle: f32, // 0 for center, -angle for left, +angle for right
    sensor_distance: f32,
    boundary_rect: &Rect<u32>,
) -> Point2 {
    let sensor_abs_heading = rotate_by_degrees(agent_heading, sensor_relative_angle);
    let sensor_abs_heading_rad = sensor_abs_heading.to_radians();

    let world_width = boundary_rect.x_max() as f32;
    let world_height = boundary_rect.y_max() as f32;

    // Calculate unclamped sensor position
    let unclamped_x = agent_location.x + sensor_distance * sensor_abs_heading_rad.sin();
    let unclamped_y = agent_location.y + sensor_distance * sensor_abs_heading_rad.cos();

    // Wrap coordinates
    // The modulo operator % in Rust behaves like a remainder, so for negative numbers,
    // we add world_width/height before the second modulo to ensure a positive result.
    let wrapped_x = (unclamped_x % world_width + world_width) % world_width;
    let wrapped_y = (unclamped_y % world_height + world_height) % world_height;

    Point2 {
        x: wrapped_x,
        y: wrapped_y,
    }
}

pub fn move_in_direction_of_heading(
    location: &mut Point2,
    heading: f32,
    speed: f32,
    delta_t: f32,
    boundary_rect: &Rect<u32>,
) {
    let heading_in_radians = heading.to_radians();

    move_relative_clamping(
        location,
        delta_t * speed * heading_in_radians.sin(),
        delta_t * speed * heading_in_radians.cos(),
        boundary_rect,
    );
}

#[allow(dead_code)]
pub fn move_relative_wrapping(xy: &mut Point2, x: f32, y: f32, boundary_rect: &Rect<u32>) {
    xy.x += x;
    xy.y += y;

    if xy.x >= boundary_rect.x_max() as f32 {
        xy.x -= boundary_rect.x_max() as f32;
    } else if xy.x < boundary_rect.x_min() as f32 {
        xy.x += boundary_rect.x_max() as f32;
    }

    if xy.y >= boundary_rect.y_max() as f32 {
        xy.y -= boundary_rect.y_max() as f32;
    } else if xy.y < boundary_rect.y_min() as f32 {
        xy.y += boundary_rect.y_max() as f32;
    }
}

pub fn move_relative_clamping(xy: &mut Point2, x: f32, y: f32, boundary_rect: &Rect<u32>) {
    xy.x += x;
    xy.y += y;

    if xy.x > boundary_rect.x_max() as f32 {
        xy.x = boundary_rect.x_max() as f32;
    } else if xy.x < boundary_rect.x_min() as f32 {
        xy.x = boundary_rect.x_min() as f32;
    }

    if xy.y > boundary_rect.y_max() as f32 {
        xy.y = boundary_rect.y_max() as f32;
    } else if xy.y < boundary_rect.y_min() as f32 {
        xy.y = boundary_rect.y_min() as f32;
    }
}

fn default_rng() -> StdRng {
    StdRng::from_os_rng()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rotate_by_degrees_handles_clockwise_rotations_correctly() {
        let rotation_amount = 45.0;
        let start_rotation = 0.0;
        let expected_end_rotation = 45.0;
        let actual_end_rotation = rotate_by_degrees(start_rotation, rotation_amount);

        assert_eq!(expected_end_rotation, actual_end_rotation)
    }

    #[test]
    fn rotate_by_degrees_handles_clockwise_rotations_correctly_wrapping() {
        let rotation_amount = 20.0;
        let start_rotation = 350.0;
        let expected_end_rotation = 10.0;
        let actual_end_rotation = rotate_by_degrees(start_rotation, rotation_amount);

        assert_eq!(expected_end_rotation, actual_end_rotation)
    }

    #[test]
    fn rotate_by_degrees_handles_clockwise_rotations_correctly_big_number() {
        let rotation_amount = 6781350.0;
        let start_rotation = 350.0;
        let expected_end_rotation = 20.0;
        let actual_end_rotation = rotate_by_degrees(start_rotation, rotation_amount);

        assert_eq!(expected_end_rotation, actual_end_rotation)
    }

    #[test]
    fn rotate_by_degrees_handles_counterclockwise_rotations_correctly() {
        let rotation_amount = -45.0;
        let start_rotation = 0.0;
        let expected_end_rotation = 315.0;
        let actual_end_rotation = rotate_by_degrees(start_rotation, rotation_amount);

        assert_eq!(expected_end_rotation, actual_end_rotation)
    }

    #[test]
    fn rotate_by_degrees_handles_counterclockwise_rotations_correctly_wrapping() {
        let rotation_amount = -20.0;
        let start_rotation = 10.0;
        let expected_end_rotation = 350.0;
        let actual_end_rotation = rotate_by_degrees(start_rotation, rotation_amount);

        assert_eq!(expected_end_rotation, actual_end_rotation)
    }

    #[test]
    fn rotate_by_degrees_handles_counterclockwise_rotations_correctly_big_number() {
        let rotation_amount = -13246790.0;
        let start_rotation = 10.0;
        let expected_end_rotation = 140.0;
        let actual_end_rotation = rotate_by_degrees(start_rotation, rotation_amount);

        assert_eq!(expected_end_rotation, actual_end_rotation)
    }
}
