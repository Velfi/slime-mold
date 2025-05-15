use crate::{Agent, Point2, Swapper};
use log::{debug, trace};
use nalgebra::DMatrix;

pub type StaticGradientGeneratorFn = Box<dyn Fn(u32, u32) -> Vec<f32>>;

pub struct Pheromones {
    grid: Swapper<DMatrix<f32>>,
    static_gradient: Option<DMatrix<f32>>,
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
        static_gradient_generator: Option<StaticGradientGeneratorFn>,
    ) -> Self {
        let grid_matrix = DMatrix::zeros(height as usize, width as usize);
        let mut static_gradient_matrix = None;
        let mut enable_static_gradient = false;

        if let Some(generator) = &static_gradient_generator {
            let sg_vec = generator(width, height);
            if sg_vec.len() == (width * height) as usize {
                static_gradient_matrix =
                    Some(DMatrix::from_vec(height as usize, width as usize, sg_vec));
                enable_static_gradient = true;
            } else {
                debug!(
                    "Static gradient generator produced a Vec of incorrect size. Expected {}, got {}.",
                    width * height,
                    sg_vec.len()
                );
            }
        }

        debug!(
            "Created new DMatrix grid with height {} and width {}",
            height, width
        );

        let grid = Swapper::new(grid_matrix.clone(), grid_matrix);

        Self {
            decay_factor,
            enable_static_gradient,
            enable_dynamic_gradient,
            grid,
            static_gradient: static_gradient_matrix,
            height,
            width,
        }
    }

    pub fn static_gradient(&self) -> Option<&DMatrix<f32>> {
        self.static_gradient.as_ref()
    }

    pub fn get_reading(&self, at_location: Point2) -> Option<i32> {
        let (x, y) = (at_location.x.round(), at_location.y.round());
        let is_within_bounds =
            x >= 0.0 && x < self.width as f32 && y >= 0.0 && y < self.height as f32;

        if is_within_bounds {
            let (x_u32, y_u32) = (x as u32, y as u32);
            let row = y_u32 as usize;
            let col = x_u32 as usize;

            match (self.enable_dynamic_gradient, self.enable_static_gradient) {
                (true, true) => {
                    let pheromone_value = self.grid.a()[(row, col)];
                    let static_value = self
                        .static_gradient
                        .as_ref()
                        .map(|grid_data| grid_data[(row, col)]);

                    static_value.map(|sv| ((sv + pheromone_value) * 255.0).round() as i32)
                }
                (true, false) => Some((self.grid.a()[(row, col)] * 255.0).round() as i32),
                (false, true) => self
                    .static_gradient
                    .as_ref()
                    .map(|grid_data| (grid_data[(row, col)] * 255.0).round() as i32),
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
        if !self.enable_dynamic_gradient {
            return;
        }

        let (current_pheromones_data, next_pheromones_data) = self.grid.read_a_write_b();

        *next_pheromones_data = current_pheromones_data.clone();

        for agent in agents {
            let location_to_deposit: Point2 = agent.location();
            let x_f32 = location_to_deposit.x;
            let y_f32 = location_to_deposit.y;

            // Wrap coordinates
            let x_wrapped = (x_f32 % self.width as f32 + self.width as f32) % self.width as f32;
            let y_wrapped = (y_f32 % self.height as f32 + self.height as f32) % self.height as f32;

            let x = x_wrapped as u32;
            let y = y_wrapped as u32;
            let row = y as usize;
            let col = x as usize;

            if row < next_pheromones_data.nrows() && col < next_pheromones_data.ncols() {
                next_pheromones_data[(row, col)] = agent.deposition_amount();
            } else {
                trace!(
                    "Calculated index ({}, {}) out of bounds for DMatrix with shape ({}, {}) during deposition",
                    row,
                    col,
                    next_pheromones_data.nrows(),
                    next_pheromones_data.ncols()
                );
            }
        }
        self.grid.swap();
    }

    pub fn diffuse(&mut self) {
        if !self.enable_dynamic_gradient {
            return;
        }
        let (grid_a_data, grid_b_data) = self.grid.read_a_write_b();
        *grid_b_data = box_filter(grid_a_data, self.width, self.height, 1, 1);
        self.grid.swap()
    }

    pub fn decay(&mut self) {
        if !self.enable_dynamic_gradient {
            return;
        }

        let (current_pheromones_data, next_pheromones_data) = self.grid.read_a_write_b();
        let decay_factor = self.decay_factor;

        *next_pheromones_data = current_pheromones_data * (1.0 - decay_factor);

        self.grid.swap();
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

    pub fn get_current_grid(&self) -> &DMatrix<f32> {
        self.grid.a()
    }
}

fn box_filter(
    image_data: &DMatrix<f32>,
    width: u32,
    height: u32,
    x_radius: u32,
    y_radius: u32,
) -> DMatrix<f32> {
    let matrix_rows = image_data.nrows();
    let matrix_cols = image_data.ncols();

    let mut out_data = DMatrix::zeros(matrix_rows, matrix_cols);

    let kernel_width = 2 * x_radius + 1;
    let kernel_height = 2 * y_radius + 1;
    let kernel_size = (kernel_width * kernel_height) as f32;

    for y_out_u in 0..height {
        for x_out_u in 0..width {
            let y_out = y_out_u as usize;
            let x_out = x_out_u as usize;
            let mut sum = 0.0;
            for ky in 0..kernel_height {
                for kx in 0..kernel_width {
                    let x_in_intermediate = x_out_u as i32 + kx as i32 - x_radius as i32;
                    let y_in_intermediate = y_out_u as i32 + ky as i32 - y_radius as i32;

                    // Wrap coordinates
                    let x_in = (x_in_intermediate % width as i32 + width as i32) % width as i32;
                    let y_in = (y_in_intermediate % height as i32 + height as i32) % height as i32;

                    sum += image_data[(y_in as usize, x_in as usize)];
                }
            }
            let avg = sum / kernel_size;
            out_data[(y_out, x_out)] = avg;
        }
    }
    out_data
}
