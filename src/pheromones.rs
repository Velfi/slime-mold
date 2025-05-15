#![allow(dead_code)]

use crate::{Agent, Point2, Swapper};
use log::{debug, trace};

pub struct Pheromones {
    grid: Swapper<Vec<f32>>,
    static_gradient: Option<Vec<f32>>,
    enable_static_gradient: bool,
    enable_dynamic_gradient: bool,
    decay_factor: f32,
    height: u32,
    width: u32,
}

impl Pheromones {
    pub fn new(
        width: u32,
        height: u32,
        decay_factor: f32,
        enable_dynamic_gradient: bool,
        static_gradient_generator: Option<Box<dyn Fn(u32, u32) -> Vec<f32>>>, // Changed return type
    ) -> Self {
        let grid_data = vec![0.0f32; (width * height) as usize]; // Initialize Vec<f32>
        let mut static_gradient = None;
        let mut enable_static_gradient = false;

        if let Some(generator) = &static_gradient_generator {
            static_gradient = Some(generator(width, height));
            enable_static_gradient = true;
        }

        debug!(
            "Created new grid with width {} and height {}",
            width, height
        );

        let grid = Swapper::new(grid_data.clone(), grid_data);

        Self {
            decay_factor,
            enable_static_gradient,
            enable_dynamic_gradient,
            grid,
            static_gradient,
            height,
            width,
        }
    }

    pub fn static_gradient(&self) -> Option<&Vec<f32>> {
        // Changed return type
        self.static_gradient.as_ref()
    }

    // TODO make this a wrapping get
    pub fn get_reading(&self, at_location: Point2) -> Option<i32> {
        let (x, y) = (at_location.x.round(), at_location.y.round());
        let is_within_bounds =
            x >= 0.0 && x < self.width as f32 && y >= 0.0 && y < self.height as f32;

        if is_within_bounds {
            let (x_u32, y_u32) = (x as u32, y as u32);
            let index = (y_u32 * self.width + x_u32) as usize;

            match (self.enable_dynamic_gradient, self.enable_static_gradient) {
                (true, true) => {
                    let pheromone_value = self.grid.a()[index];
                    let static_value = self
                        .static_gradient
                        .as_ref()
                        .map(|grid_data| grid_data[index]);

                    static_value.map(|sv| ((sv + pheromone_value) * 255.0).round() as i32)
                }
                (true, false) => Some((self.grid.a()[index] * 255.0).round() as i32),
                (false, true) => self
                    .static_gradient
                    .as_ref()
                    .map(|grid_data| (grid_data[index] * 255.0).round() as i32),
                (false, false) => Some(0),
            }
        } else {
            None
        }
    }

    pub fn len(&self) -> u32 {
        self.width * self.height
    }

    pub fn deposit(&mut self, agents: &[Agent]) {
        agents
            .iter()
            .for_each(|agent| self.deposit_individual_agent(agent))
    }

    fn deposit_individual_agent(&mut self, agent: &Agent) {
        let location_to_deposit: Point2 = agent.location();
        let x_f32 = location_to_deposit.x;
        let y_f32 = location_to_deposit.y;

        let is_within_bounds =
            x_f32 >= 0.0 && x_f32 < self.width as f32 && y_f32 >= 0.0 && y_f32 < self.height as f32;

        if is_within_bounds {
            let x = x_f32 as u32;
            let y = y_f32 as u32;
            let index = (y * self.width + x) as usize;

            self.grid.mut_a()[index] = agent.deposition_amount();
        } else {
            trace!("agent out of bounds at ({})", location_to_deposit)
        }
    }

    pub fn diffuse(&mut self) {
        let (grid_a_data, grid_b_data) = self.grid.read_a_write_b();
        *grid_b_data = box_filter(grid_a_data, self.width, self.height, 1, 1); // Pass dimensions
        self.grid.swap()
    }

    pub fn decay(&mut self) {
        let decay_factor = self.decay_factor;
        self.grid.mut_a().iter_mut().for_each(|pheromone_reading| {
            *pheromone_reading = *pheromone_reading * (1.0 - decay_factor);
        })
    }

    pub fn set_decay_factor(&mut self, decay_factor: f32) {
        self.decay_factor = decay_factor;
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.grid.a().iter()
    }

    pub fn enable_dynamic_gradient(&mut self) {
        self.enable_dynamic_gradient = true;
    }

    pub fn disable_dynamic_gradient(&mut self) {
        self.enable_dynamic_gradient = false;
    }
}

fn box_filter(
    image_data: &[f32],
    width: u32,
    height: u32,
    x_radius: u32,
    y_radius: u32,
) -> Vec<f32> {
    let mut out_data = vec![0.0f32; (width * height) as usize];

    let kernel_width = 2 * x_radius + 1;
    let kernel_height = 2 * y_radius + 1;
    let kernel_size = (kernel_width * kernel_height) as f32;

    for y_out in 0..height {
        for x_out in 0..width {
            let mut sum = 0.0;
            for ky in 0..kernel_height {
                for kx in 0..kernel_width {
                    let x_in = (x_out as i32 + kx as i32 - x_radius as i32)
                        .max(0)
                        .min(width as i32 - 1) as u32;
                    let y_in = (y_out as i32 + ky as i32 - y_radius as i32)
                        .max(0)
                        .min(height as i32 - 1) as u32;

                    let index_in = (y_in * width + x_in) as usize;
                    sum += image_data[index_in];
                }
            }
            let avg = sum / kernel_size;
            let index_out = (y_out * width + x_out) as usize;
            out_data[index_out] = avg;
        }
    }
    out_data
}

pub trait StaticGradientGenerator {
    fn generate(width: u32, height: u32) -> Vec<f32>;
}

// TODO only works if height == width
pub fn generate_circular_static_gradient(width: u32, height: u32) -> Vec<f32> {
    let min_value: f32 = 0.0;
    let max_value: f32 = 1.0; // Changed max_value to 1.0 for f32 representation
    let root_2 = 2.0f32.sqrt();

    let vec: Vec<_> = (0..width)
        .flat_map(|x_coord| (0..height).map(move |y_coord| (x_coord as f32, y_coord as f32)))
        .map(|(x_f, y_f)| {
            let w_f = width as f32;
            let h_f = height as f32;
            let mut distance_to_center =
                ((x_f - w_f / 2.0).powi(2) + (y_f - h_f / 2.0).powi(2)).sqrt();

            distance_to_center /= root_2 * w_f / 2.0;

            let t = min_value * distance_to_center + max_value * (1.0 - distance_to_center);

            t
        })
        .collect();

    assert!(
        vec.len() == (width * height) as usize,
        "Vector length is {} but width * height is {}",
        vec.len(),
        width * height
    );

    vec
}

// TODO replace custom gradient generation with raqote?
pub fn generate_linear_static_gradient(width: u32, height: u32) -> Vec<f32> {
    let min_value: f32 = 0.0;
    let max_value: f32 = 1.0; // Changed max_value to 1.0 for f32 representation
    let a: f32 = -0.6;
    let b: f32 = -1.0;
    let c: f32 = width as f32 - (width as f32 / 4.0);

    let vec: Vec<_> = (0..width)
        .flat_map(|x_coord| (0..height).map(move |y_coord| (x_coord as f32, y_coord as f32)))
        .map(|(x_f, y_f)| {
            let w_f = width as f32;

            let distance = (a * x_f + b * y_f + c) / (a * a + b * b).sqrt();
            let color_coef = distance.abs() / w_f;

            let t = min_value * color_coef + max_value * (1.0 - color_coef);

            t
        })
        .collect();

    assert!(
        vec.len() == (width * height) as usize,
        "Vector length is {} but width * height is {}",
        vec.len(),
        width * height
    );

    vec
}
