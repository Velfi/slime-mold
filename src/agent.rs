use crate::{Pheromones, Point2, DEPOSITION_AMOUNT, HEIGHT, WIDTH};
use log::trace;
use num::{Float, NumCast};
use rand::Rng;
use typed_builder::TypedBuilder;

pub type SensorReading = (f64, f64, f64);

#[derive(TypedBuilder)]
pub struct Agent {
    location: Point2<f64>,
    // The heading and agent is facing. (In degrees)
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
    #[builder(default)]
    rng: fastrand::Rng,
    #[builder(default = DEPOSITION_AMOUNT)]
    deposition_amount: f64,
}

pub struct AgentUpdate {
    pub location: Option<Point2<f64>>,
    pub heading: Option<f64>,
    pub sensor_angle: Option<f64>,
    pub sensor_distance: Option<f64>,
    pub move_speed: Option<f64>,
    pub rotation_speed: Option<f64>,
    pub jitter: Option<f64>,
    pub deposition_amount: Option<f64>,
}

impl Default for AgentUpdate {
    fn default() -> Self {
        AgentUpdate {
            location: None,
            heading: None,
            sensor_angle: None,
            sensor_distance: None,
            move_speed: None,
            rotation_speed: None,
            jitter: None,
            deposition_amount: None,
        }
    }
}

impl Agent {
    pub fn judge_sensory_input(
        &self,
        (l_reading, c_reading, r_reading): SensorReading,
        rng: &mut impl Rng,
    ) -> f64 {
        if c_reading > l_reading && c_reading > r_reading {
            // do nothing, stay facing same direction
            trace!("Agent's center value is greatest, doing nothing");
            0.0
        } else if c_reading < l_reading && c_reading < r_reading {
            // rotate randomly to the left or right
            let should_rotate_right: bool = rng.gen();

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
            trace!("Agent is choosing random rotation (final fallthrough case)");
            rng.gen_range(-self.rotation_speed..self.rotation_speed)
        }
    }

    pub fn location(&self) -> Point2<f64> {
        self.location
    }

    pub fn deposition_amount(&self) -> f64 {
        self.deposition_amount
    }

    // TODO why don't they fear the edges?
    pub fn sense(&self, pheromones: &Pheromones) -> SensorReading {
        let sensor_l_heading = rotate_by_degrees(self.heading, -self.sensor_angle);
        let sensor_c_heading = self.heading;
        let sensor_r_heading = rotate_by_degrees(self.heading, self.sensor_angle);

        let sensor_l_location =
            move_in_direction_of_heading(self.location, sensor_l_heading, self.sensor_distance);
        let sensor_c_location =
            move_in_direction_of_heading(self.location, sensor_c_heading, self.sensor_distance);
        let sensor_r_location =
            move_in_direction_of_heading(self.location, sensor_r_heading, self.sensor_distance);

        // This assumes that there is a 1:1 relationship between an agent's possible
        // movement space in a grid, and the pheromone field. What if we want to have agents moving
        // around and storing that at one level of detail and save the pheromone field at another level
        // of detail?
        // Also, if a sensor goes out of bounds, it just reads 0.0
        // Maybe it'd be better to wrap the agents (and their sensors) to the other side of the field?
        let sensor_l_reading = pheromones.get_reading(sensor_l_location).unwrap_or(-1.0);
        let sensor_c_reading = pheromones.get_reading(sensor_c_location).unwrap_or(-1.0);
        let sensor_r_reading = pheromones.get_reading(sensor_r_location).unwrap_or(-1.0);

        (sensor_l_reading, sensor_c_reading, sensor_r_reading)
    }

    pub fn rotate(&mut self, mut rotation_in_degrees: f64) {
        if self.jitter != 0.0 {
            // Randomly adjust rotation amount
            rotation_in_degrees += self.jitter * self.rng.f64();
        }

        self.heading = rotate_by_degrees(self.heading, rotation_in_degrees);
        trace!("new heading is {}", self.heading);
    }

    pub fn move_in_direction_of_current_heading(&mut self) {
        let heading_in_radians = self.heading.to_radians();
        self.location.move_relative(
            self.move_speed * heading_in_radians.sin(),
            self.move_speed * heading_in_radians.cos(),
        );

        self.location
            .clamp(0.0, 0.0, (WIDTH - 1) as f64, (HEIGHT - 1) as f64)
    }

    pub fn apply_update(&mut self, update: &AgentUpdate) {
        update.location.and_then(|val| Some(self.location = val));
        update.heading.and_then(|val| Some(self.heading = val));
        update
            .sensor_angle
            .and_then(|val| Some(self.sensor_angle = val));
        update
            .sensor_distance
            .and_then(|val| Some(self.sensor_distance = val));
        update
            .move_speed
            .and_then(|val| Some(self.move_speed = val));
        update
            .rotation_speed
            .and_then(|val| Some(self.rotation_speed = val));
        update.jitter.and_then(|val| Some(self.jitter = val));
        update
            .deposition_amount
            .and_then(|val| Some(self.deposition_amount = val));
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

fn move_in_direction_of_heading(
    mut location: Point2<f64>,
    heading: f64,
    distance: f64,
) -> Point2<f64> {
    let heading_in_radians = heading.to_radians();

    location.move_relative(
        distance * heading_in_radians.sin(),
        distance * heading_in_radians.cos(),
    );

    location
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

    // When doing math on floating point numbers, we inevitably run into inaccuracies.
    // We allow for a small margin of error in order to avoid failing a test on account
    // of those small, expected inaccuracies.
    const FLOAT_MARGIN_OF_ERROR: f64 = 0.001;

    #[test]
    fn move_in_direction_of_heading_if_no_movement() {
        let start_location = Point2::new(0.0f64, 0.0);
        let heading = 0.0f64;
        let distance = 0.0f64;
        let expected_end_location = Point2::new(0.0f64, 0.0);
        let actual_end_location = move_in_direction_of_heading(start_location, heading, distance);

        assert!(expected_end_location.distance_to(&actual_end_location) < FLOAT_MARGIN_OF_ERROR)
    }

    #[test]
    fn move_in_direction_of_heading_if_move_north_one_unit() {
        let start_location = Point2::new(0.0f64, 0.0);
        let heading = 0.0f64;
        let distance = 1.0f64;
        let expected_end_location = Point2::new(0.0f64, 1.0);
        let actual_end_location = move_in_direction_of_heading(start_location, heading, distance);

        assert!(expected_end_location.distance_to(&actual_end_location) < FLOAT_MARGIN_OF_ERROR)
    }

    #[test]
    fn move_in_direction_of_heading_if_move_east_one_unit() {
        let start_location = Point2::new(0.0f64, 0.0);
        let heading = 90.0f64;
        let distance = 1.0f64;
        let expected_end_location = Point2::new(1.0f64, 0.0);
        let actual_end_location = move_in_direction_of_heading(start_location, heading, distance);

        assert!(expected_end_location.distance_to(&actual_end_location) < FLOAT_MARGIN_OF_ERROR)
    }

    #[test]
    fn move_in_direction_of_heading_if_move_south_one_unit() {
        let start_location = Point2::new(0.0f64, 0.0);
        let heading = 180.0f64;
        let distance = 1.0f64;
        let expected_end_location = Point2::new(0.0f64, -1.0);
        let actual_end_location = move_in_direction_of_heading(start_location, heading, distance);

        assert!(expected_end_location.distance_to(&actual_end_location) < FLOAT_MARGIN_OF_ERROR)
    }

    #[test]
    fn move_in_direction_of_heading_if_move_west_one_unit() {
        let start_location = Point2::new(0.0f64, 0.0);
        let heading = 270.0f64;
        let distance = 1.0f64;
        let expected_end_location = Point2::new(-1.0f64, 0.0);
        let actual_end_location = move_in_direction_of_heading(start_location, heading, distance);

        assert!(expected_end_location.distance_to(&actual_end_location) < FLOAT_MARGIN_OF_ERROR)
    }

    #[test]
    fn move_in_direction_of_heading_treats_equivalent_headings_equally() {
        let location_from_positive_heading = {
            let start_location = Point2::new(0.0f64, 0.0);
            let heading = 270.0f64;
            let distance = 1.0f64;
            move_in_direction_of_heading(start_location, heading, distance)
        };

        let location_from_negative_heading = {
            let start_location = Point2::new(0.0f64, 0.0);
            let heading = -90.0f64;
            let distance = 1.0f64;
            move_in_direction_of_heading(start_location, heading, distance)
        };

        assert!(
            location_from_positive_heading.distance_to(&location_from_negative_heading)
                < FLOAT_MARGIN_OF_ERROR
        )
    }

    // #[test]
    // fn move_in_direction_of_heading_if_location_is_zero() {
    //     todo!()
    // }
}
