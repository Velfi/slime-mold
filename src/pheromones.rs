use crate::{Agent, Point2, Swapper};
use image::{io::Reader as ImageReader, GrayImage};
use imageproc::filter::*;
use log::{debug, trace};
use std::path::Path;

pub struct Pheromones {
    grid: Swapper<GrayImage>,
    static_gradient: Option<GrayImage>,
    enable_static_gradient: bool,
    enable_dynamic_gradient: bool,
    decay_factor: u8,
    _deposit_size: f64,
    _diffuse_size: f64,
    height: u32,
    width: u32,
}

impl Pheromones {
    pub fn new(
        width: u32,
        height: u32,
        decay_factor: u8,
        enable_dynamic_gradient: bool,
        static_gradient_generator: Option<Box<dyn Fn(u32, u32) -> GrayImage>>,
    ) -> Self {
        let grid = GrayImage::new(width, height);
        let mut static_gradient = None;
        let mut enable_static_gradient = false;

        if let Some(generator) = &static_gradient_generator {
            static_gradient = Some(generator(width, height));
            enable_static_gradient = true;
        }

        debug!(
            "Created new grid with width {} and height {}",
            grid.width(),
            grid.height()
        );

        let grid = Swapper::new(grid.clone(), grid);

        Self {
            _deposit_size: 1.0,
            _diffuse_size: 3.0,
            decay_factor,
            enable_static_gradient,
            enable_dynamic_gradient,
            grid,
            static_gradient,
            height,
            width,
        }
    }

    pub fn static_gradient(&self) -> Option<&GrayImage> {
        self.static_gradient.as_ref()
    }

    // TODO make this a wrapping get
    pub fn get_reading(&self, at_location: Point2<f64>) -> Option<i32> {
        let (x, y) = (at_location.x.round(), at_location.y.round());
        let is_within_bounds =
            x >= 0.0 && x < self.width as f64 && y >= 0.0 && y < self.height as f64;

        if is_within_bounds {
            let (x, y) = (x as u32, y as u32);

            match (self.enable_dynamic_gradient, self.enable_static_gradient) {
                (true, true) => {
                    let pheromone_value = self.grid.a().get_pixel(x, y);
                    let static_value = self
                        .static_gradient
                        .as_ref()
                        .map(|grid| grid.get_pixel(x, y));

                    static_value.map(|sv| (sv.0[0] + pheromone_value.0[0]) as i32)
                }
                (true, false) => Some(self.grid.a().get_pixel(x, y).0[0] as i32),
                (false, true) => self
                    .static_gradient
                    .as_ref()
                    .map(|grid| grid.get_pixel(x, y).0[0] as i32),
                (false, false) => Some(0),
            }
        } else {
            None
        }
    }

    pub fn len(&self) -> u32 {
        self.grid.a().width() * self.grid.a().height()
    }

    pub fn deposit(&mut self, agents: &[Agent]) {
        agents
            .iter()
            .for_each(|agent| self.deposit_individual_agent(agent))
    }

    fn deposit_individual_agent(&mut self, agent: &Agent) {
        let location_to_deposit: Point2<u32> = agent.location().into();
        let is_within_bounds =
            location_to_deposit.x < self.width && location_to_deposit.y < self.height;

        if is_within_bounds {
            let grid_ref = self
                .grid
                .mut_a()
                .get_pixel_mut(location_to_deposit.x, location_to_deposit.y);

            *grid_ref = [agent.deposition_amount()].into();
        } else {
            trace!("agent out of bounds at ({})", location_to_deposit)
        }
    }

    pub fn diffuse(&mut self) {
        let (grid_a, grid_b) = self.grid.read_a_write_b();
        *grid_b = box_filter(&grid_a, 1, 1);
        self.grid.swap()
    }

    pub fn decay(&mut self) {
        let decay_factor = self.decay_factor;
        self.grid.mut_a().iter_mut().for_each(|pheromone_reading| {
            *pheromone_reading = pheromone_reading.saturating_sub(decay_factor)
        })
    }

    pub fn set_decay_factor(&mut self, decay_factor: u8) {
        self.decay_factor = decay_factor;
    }

    pub fn iter(&self) -> impl Iterator<Item = &u8> {
        self.grid.a().iter()
    }

    pub fn enable_dynamic_gradient(&mut self) {
        self.enable_dynamic_gradient = true;
    }

    pub fn disable_dynamic_gradient(&mut self) {
        self.enable_dynamic_gradient = false;
    }
}

pub trait StaticGradientGenerator {
    fn generate(width: u32, height: u32) -> GrayImage;
}

// TODO only works if height == width
pub fn generate_circular_static_gradient(width: u32, height: u32) -> GrayImage {
    let min_value: f64 = 0.0;
    let max_value: f64 = 255.0;
    let root_2 = 2.0f64.sqrt();

    let vec: Vec<_> = (0..width)
        .into_iter()
        .map(|x| (0..height).into_iter().map(move |y| (x as f64, y as f64)))
        .flatten()
        .map(|(x, y)| {
            let width = width as f64;
            let height = height as f64;
            let mut distance_to_center =
                ((x - width / 2.0).powi(2) + (y - height / 2.0).powi(2)).sqrt();

            distance_to_center = distance_to_center / (root_2 * width / 2.0);

            let t = min_value * distance_to_center + max_value * (1.0 - distance_to_center);

            t.round() as u8
        })
        .collect();

    assert!(
        vec.len() == (width * height) as usize,
        "Vector length is {} but width * height is {}",
        vec.len(),
        width * height
    );

    GrayImage::from_raw(width, height, vec).unwrap()
}

// TODO replace custom gradient generation with raqote?
pub fn generate_linear_static_gradient(width: u32, height: u32) -> GrayImage {
    let min_value: f64 = 0.0;
    let max_value: f64 = 255.0;
    let a: f64 = -0.6;
    let b: f64 = -1.0;
    let c: f64 = width as f64 - (width as f64 / 4.0);

    let vec: Vec<_> = (0..width)
        .into_iter()
        .map(|x| (0..height).into_iter().map(move |y| (x as f64, y as f64)))
        .flatten()
        .map(|(x, y)| {
            let width = width as f64;

            let distance = (a * x + b * y + c) / (a * a + b * b).sqrt();
            let color_coef = distance.abs() / width;

            let t = min_value * color_coef + max_value * (1.0 - color_coef);

            t.round() as u8
        })
        .collect();

    assert!(
        vec.len() == (width * height) as usize,
        "Vector length is {} but width * height is {}",
        vec.len(),
        width * height
    );

    GrayImage::from_raw(width, height, vec).unwrap()
}

// Warp this in a closure in order to pass it in when building a `Pheromones`
// e.g. `|width: u32, height: u32| pheromones::generate_image_based_static_gradient(width, height, "images/slime.png")`
pub fn generate_image_based_static_gradient(
    width: u32,
    height: u32,
    image_path: impl AsRef<Path>,
) -> GrayImage {
    let river_img = ImageReader::open(image_path)
        .expect("loading image file")
        .decode()
        .expect("decoding image file");
    let resized_img = river_img.resize(width, height, image::imageops::FilterType::Gaussian);

    resized_img.into_luma8()
}
