use std::ops::Range;

use crate::{Pheromones, Point2, rect::Rect, settings::Settings};
use log::trace;
use num::{Float, NumCast};
use rand::prelude::*;
use typed_builder::TypedBuilder;

pub type SensorReading = (i32, i32, i32);

#[derive(TypedBuilder)]
pub struct Agent {
    location: Point2<f64>,
    // The heading an agent is facing. (In degrees)
    #[builder(default)]
    heading: f64,
    // There are three sensors per agent: A center sensor, a left sensor, and a right sensor. The side sensors are positioned based on this angle. (In degrees)
    #[builder(default = 45.0f64)]
    sensor_angle: f64,
    // How far out a sensor is from the agent
    #[builder(default = 9.0f64)]
    sensor_distance: f64,
    // How far out a sensor is from the agent
    #[builder(default = 1.0f64)]
    move_speed: f64,
    // How quickly the agent can rotate
    #[builder(default = 20.0f64)]
    rotation_speed: f64,
    // The tendency of agents to move erratically
    #[builder(default = 0.0f64)]
    jitter: f64,
    #[builder(default = SeedableRng::from_entropy())]
    rng: StdRng,
    #[builder()]
    deposition_amount: u8,
}

#[derive(Default)]
pub struct AgentUpdate {
    pub location: Option<Point2<f64>>,
    pub heading: Option<f64>,
    pub sensor_angle: Option<f64>,
    pub sensor_distance: Option<f64>,
    pub move_speed: Option<f64>,
    pub rotation_speed: Option<f64>,
    pub jitter: Option<f64>,
    pub deposition_amount: Option<u8>,
}

impl Agent {
    pub fn new_from_settings(settings: &Settings) -> Self {
        let mut rng: StdRng = SeedableRng::from_entropy();
        let deposition_amount = settings.agent_deposition_amount;
        let move_speed = rng.gen_range(settings.agent_speed_min..settings.agent_speed_max);
        let location = Point2::new(
            rng.gen_range(0.0..(settings.window_width as f64)),
            rng.gen_range(0.0..(settings.window_height as f64)),
        );
        let heading = rng.gen_range(settings.agent_possible_starting_headings.clone());

        Agent::builder()
            .location(location)
            .heading(heading)
            .move_speed(move_speed)
            .jitter(settings.agent_jitter)
            .deposition_amount(deposition_amount)
            .rotation_speed(settings.agent_turn_speed)
            .rng(SeedableRng::from_entropy())
            .build()
    }

    pub fn update(&mut self, pheromones: &Pheromones, delta_t: f64, boundary_rect: &Rect<u32>) {
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

    pub fn judge_sensory_input(&mut self, (l_reading, c_reading, r_reading): SensorReading) -> f64 {
        if c_reading > l_reading && c_reading > r_reading {
            // do nothing, stay facing same direction
            trace!("Agent's center value is greatest, doing nothing");
            0.0
        } else if c_reading < l_reading && c_reading < r_reading {
            // rotate randomly to the left or right
            let should_rotate_right: bool = self.rng.r#gen();

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

    pub fn location(&self) -> Point2<f64> {
        self.location
    }

    pub fn deposition_amount(&self) -> u8 {
        self.deposition_amount
    }

    pub fn set_new_random_move_speed_in_range(&mut self, move_speed_range: Range<f64>) {
        self.move_speed = self.rng.gen_range(move_speed_range);
        trace!("set agent's speed to {}", self.move_speed);
    }

    // TODO why don't they fear the edges?
    pub fn sense(&self, pheromones: &Pheromones, boundary_rect: &Rect<u32>) -> SensorReading {
        let mut sensor_l_location = self.location;
        move_in_direction_of_heading(
            &mut sensor_l_location,
            rotate_by_degrees(self.heading, -self.sensor_angle),
            self.sensor_distance,
            1.0,
            boundary_rect,
        );
        let mut sensor_c_location = self.location;
        move_in_direction_of_heading(
            &mut sensor_c_location,
            self.heading,
            self.sensor_distance,
            1.0,
            boundary_rect,
        );
        let mut sensor_r_location = self.location;
        move_in_direction_of_heading(
            &mut sensor_r_location,
            rotate_by_degrees(self.heading, self.sensor_angle),
            self.sensor_distance,
            1.0,
            boundary_rect,
        );

        // This assumes that there is a 1:1 relationship between an agent's possible
        // movement space in a grid, and the pheromone field. What if we want to have agents moving
        // around and storing that at one level of detail and save the pheromone field at another level
        // of detail?
        // Also, if a sensor goes out of bounds, it reads -1
        // Maybe it'd be better to wrap the agents (and their sensors) to the other side of the field?
        let sensor_l_reading = pheromones.get_reading(sensor_l_location).unwrap_or(0);
        let sensor_c_reading = pheromones.get_reading(sensor_c_location).unwrap_or(0);
        let sensor_r_reading = pheromones.get_reading(sensor_r_location).unwrap_or(0);

        (sensor_l_reading, sensor_c_reading, sensor_r_reading)
    }

    pub fn rotate(&mut self, mut rotation_in_degrees: f64) {
        if self.jitter != 0.0 {
            let magnitude = if self.rng.r#gen() {
                self.rng.r#gen::<f64>()
            } else {
                self.rng.r#gen::<f64>() * -1.0
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

pub fn move_in_direction_of_heading(
    location: &mut Point2<f64>,
    heading: f64,
    speed: f64,
    delta_t: f64,
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
pub fn move_relative_wrapping(xy: &mut Point2<f64>, x: f64, y: f64, boundary_rect: &Rect<u32>) {
    xy.x += x;
    xy.y += y;

    if xy.x >= boundary_rect.x_max() as f64 {
        xy.x -= boundary_rect.x_max() as f64;
    } else if xy.x < boundary_rect.x_min() as f64 {
        xy.x += boundary_rect.x_max() as f64;
    }

    if xy.y >= boundary_rect.y_max() as f64 {
        xy.y -= boundary_rect.y_max() as f64;
    } else if xy.y < boundary_rect.y_min() as f64 {
        xy.y += boundary_rect.y_max() as f64;
    }
}

pub fn move_relative_clamping(xy: &mut Point2<f64>, x: f64, y: f64, boundary_rect: &Rect<u32>) {
    xy.x += x;
    xy.y += y;

    if xy.x > boundary_rect.x_max() as f64 {
        xy.x = boundary_rect.x_max() as f64;
    } else if xy.x < boundary_rect.x_min() as f64 {
        xy.x = boundary_rect.x_min() as f64;
    }

    if xy.y > boundary_rect.y_max() as f64 {
        xy.y = boundary_rect.y_max() as f64;
    } else if xy.y < boundary_rect.y_min() as f64 {
        xy.y = boundary_rect.y_min() as f64;
    }
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
