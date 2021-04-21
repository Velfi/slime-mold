use crate::{Agent, Point2, Swapper};
use grid::Grid;
use log::debug;

pub struct Pheromones {
    grid: Swapper<Grid<f64>>,
    static_gradient: Grid<f64>,
    depositition_amount: f64,
    decay_factor: f64,
    _deposit_size: f64,
    _diffuse_size: f64,
}

const DEFAULT_DEPOSITITION_AMOUNT: f64 = 5.0;
const DEFAULT_DECAY_FACTOR: f64 = DEFAULT_DEPOSITITION_AMOUNT / 400.0;

impl Pheromones {
    pub fn new(width: usize, height: usize, init_value: f64) -> Self {
        let length = width * height;
        let init_values: Vec<_> = (0..length).map(|_| init_value).collect();
        let grid = Grid::from_vec(init_values, width);
        let static_gradient = generate_circular_static_gradient(width, height);

        debug!(
            "Created new grid with {} rows and {} columns",
            grid.rows(),
            grid.cols()
        );

        let grid = Swapper::new(grid.clone(), grid);

        Self {
            _deposit_size: 1.0,
            _diffuse_size: 3.0,
            decay_factor: DEFAULT_DECAY_FACTOR,
            depositition_amount: DEFAULT_DEPOSITITION_AMOUNT,
            grid,
            static_gradient,
        }
    }

    pub fn get_reading(&self, at_location: Point2<usize>) -> Option<f64> {
        let pheromone_value = self.grid.a().get(at_location.y(), at_location.x());
        let pheromone_gradient_value = self.static_gradient.get(at_location.y(), at_location.x());

        pheromone_value
            .map(|pv| pheromone_gradient_value.map(|pgv| pgv + pv))
            .flatten()
    }

    pub fn len(&self) -> usize {
        self.grid.a().rows() * self.grid.a().cols()
    }

    pub fn deposit(&mut self, agent: &Agent) {
        let location_to_deposit: Point2<usize> = agent.location().into();

        match self
            .grid
            .mut_a()
            .get_mut(location_to_deposit.y(), location_to_deposit.x())
        {
            Some(grid_ref) => {
                *grid_ref = self.depositition_amount;
            }
            None => {
                debug!(
                    "Tried to deposit at non-existant grid_ref {:?}",
                    location_to_deposit
                )
            }
        }
    }

    pub fn diffuse(&mut self) {
        let rows = self.grid.a().rows();
        let cols = self.grid.a().cols();
        let (grid_a, grid_b) = self.grid.read_a_write_b();

        grid_a
            .iter()
            .zip(grid_ref_iter(rows, cols))
            .for_each(|(grid_a_value, (row, col))| {
                *grid_b.get_mut(row, col).unwrap() = [
                    /* nw neighbor index */ (row.checked_sub(1), col.checked_sub(1)),
                    /* n  neighbor index */ (row.checked_sub(1), Some(col)),
                    /* ne neighbor index */ (row.checked_sub(1), col.checked_add(1)),
                    /* e  neighbor index */ (Some(row), col.checked_add(1)),
                    /* se neighbor index */ (row.checked_add(1), col.checked_add(1)),
                    /* s  neighbor index */ (row.checked_add(1), Some(col)),
                    /* sw neighbor index */ (row.checked_add(1), col.checked_sub(1)),
                    /* w  neighbor index */ (Some(row), col.checked_sub(1)),
                ]
                .iter()
                .map(|neighbor_rc| -> f64 {
                    // throw out grid indexes that are outside the grid
                    match neighbor_rc.to_owned() {
                        (Some(neighbor_row), Some(neighbor_col))
                            if neighbor_row < rows && neighbor_col < cols =>
                        {
                            *grid_b.get(neighbor_row, neighbor_col).unwrap()
                        }
                        _ => 0.0,
                    }
                })
                .fold(*grid_a_value, |mut acc, neighbor_value| {
                    acc = acc + neighbor_value;
                    acc = acc / 2.0;

                    acc
                });
            });

        self.grid.swap()
    }

    pub fn decay(&mut self) {
        let decay_factor = self.decay_factor;
        self.grid.mut_a().iter_mut().for_each(|pheromone_reading| {
            *pheromone_reading = (*pheromone_reading - decay_factor).max(0.0)
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &f64> {
        self.grid.a().iter()
    }
}

fn grid_ref_iter(rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows)
        .map(move |row| (0..cols).map(move |col| (row, col)))
        .flatten()
}

fn generate_circular_static_gradient(width: usize, height: usize) -> Grid<f64> {
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    let min_value: f64 = 0.0;
    let max_value: f64 = 1.0;
    let angle: f64 = 0.0;
    let r1: f64 = 0.0;
    let r2: f64 = width as f64;

    let vec: Vec<_> = (0..width)
        .into_iter()
        .map(|x| (0..height).into_iter().map(move |y| (x as f64, y as f64)))
        .flatten()
        .map(|(x, y)| {
            if x.powi(2) + y.powi(2) <= r2.powi(2) && x.powi(2) + y.powi(2) >= r1.powi(2) {
                let mut t = (y - center_y).atan2(x - center_x) + angle;
                // atan2 is from -pi to pi
                t = t + std::f64::consts::PI;
                // it might over 2PI becuse of +angle
                if t > 2.0 * std::f64::consts::PI {
                    t = t - 2.0 * std::f64::consts::PI
                }

                t = t / (2.0 * std::f64::consts::PI); // normalise t from 0 to 1
                (min_value * t) + (max_value * (1.0 - t))
            } else {
                0.0
            }
        })
        .collect();

    assert!(
        vec.len() == (width * height),
        "Vector length is {} but width * height is {}",
        vec.len(),
        width * height
    );

    Grid::from_vec(vec, width)
}
