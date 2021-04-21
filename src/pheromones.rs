use crate::{Agent, Point2, Swapper};
use grid::Grid;
use log::debug;

pub struct Pheromones {
    grid: Swapper<Grid<f64>>,
    static_gradient: Option<Grid<f64>>,
    static_gradient_generator: Option<Box<dyn Fn(usize, usize) -> Grid<f64>>>,
    enable_static_gradient: bool,
    enable_dynamic_gradient: bool,
    depositition_amount: f64,
    decay_factor: f64,
    _deposit_size: f64,
    _diffuse_size: f64,
    height: usize,
    width: usize,
}

const DEFAULT_DEPOSITITION_AMOUNT: f64 = 1.0;
const DEFAULT_DECAY_FACTOR: f64 = DEFAULT_DEPOSITITION_AMOUNT / 300.0;

impl Pheromones {
    pub fn new(
        width: usize,
        height: usize,
        init_value: f64,
        enable_dynamic_gradient: bool,
        static_gradient_generator: Option<Box<dyn Fn(usize, usize) -> Grid<f64>>>,
    ) -> Self {
        let length = width * height;
        let init_values: Vec<_> = (0..length).map(|_| init_value).collect();
        let grid = Grid::from_vec(init_values, width);
        let mut static_gradient = None;
        let mut enable_static_gradient = false;

        if let Some(generator) = &static_gradient_generator {
            static_gradient = Some(generator(width, height));
            enable_static_gradient = true;
        }

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
            static_gradient_generator,
            enable_static_gradient,
            enable_dynamic_gradient,
            grid,
            static_gradient,
            height,
            width,
        }
    }

    pub fn get_reading(&self, at_location: Point2<f64>) -> Option<f64> {
        let (x, y) = (at_location.x().round(), at_location.y().round());

        if x >= 0.0 && x < self.width as f64 && y >= 0.0 && y < self.height as f64 {
            let (x, y) = (x as usize, y as usize);

            match (self.enable_dynamic_gradient, self.enable_static_gradient) {
                (true, true) => {
                    let pheromone_value = self.grid.a().get(y, x).cloned();
                    let static_value = self
                        .static_gradient
                        .as_ref()
                        .map(|grid| grid.get(y as usize, x as usize).cloned())
                        .flatten();

                    pheromone_value
                        .map(|n| static_value.map(|m| n + m))
                        .flatten()
                }
                (true, false) => self.grid.a().get(y, x).cloned(),
                (false, true) => self
                    .static_gradient
                    .as_ref()
                    .map(|grid| grid.get(y as usize, x as usize).cloned())
                    .flatten(),
                (false, false) => Some(0.0),
            }
        } else {
            None
        }
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
            .zip(grid_ref_iter(cols, rows))
            .for_each(|(grid_a_value, (col, row))| {
                let sum_of_neighboring_values = neighbor_indexes(col, row, cols, rows)
                    .iter()
                    .map(|neighbor_rc| -> f64 {
                        // throw out grid indexes that are outside the grid
                        match neighbor_rc.to_owned() {
                            (Some(neighbor_col), Some(neighbor_row)) => {
                                *grid_a.get(neighbor_row, neighbor_col).unwrap()
                            }
                            _ => 0.0,
                        }
                    })
                    .sum::<f64>();

                *grid_b.get_mut(row, col).expect("invalid grid index") =
                    (grid_a_value + sum_of_neighboring_values) / 9.0;
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

fn grid_ref_iter(cols: usize, rows: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows)
        .map(move |row| (0..cols).map(move |col| (col, row)))
        .flatten()
}

fn neighbor_indexes(
    col: usize,
    row: usize,
    cols: usize,
    rows: usize,
) -> [(Option<usize>, Option<usize>); 8] {
    let rows_check = |row: usize| (row < rows).then(|| row);
    let cols_check = |col: usize| (col < cols).then(|| col);

    [
        /* nw neighbor index */ (col.checked_sub(1), row.checked_sub(1)),
        /* n  neighbor index */ (Some(col), row.checked_sub(1)),
        /* ne neighbor index */
        (col.checked_add(1).and_then(cols_check), row.checked_sub(1)),
        /* e  neighbor index */
        (col.checked_add(1).and_then(cols_check), Some(row)),
        /* se neighbor index */
        (
            col.checked_add(1).and_then(cols_check),
            row.checked_add(1).and_then(rows_check),
        ),
        /* s  neighbor index */
        (Some(col), row.checked_add(1).and_then(rows_check)),
        /* sw neighbor index */
        (col.checked_sub(1), row.checked_add(1).and_then(rows_check)),
        /* w  neighbor index */ (col.checked_sub(1), Some(row)),
    ]
}

pub trait StaticGradientGenerator {
    fn generate(width: usize, height: usize) -> Grid<f64>;
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn diffuse_does_nothing_when_pheromone_count_is_zero() {
        let mut pheromones = Pheromones::new(3, 3, 0.0, false, None);

        // Assert all cells are zero before the diffuse
        pheromones
            .iter()
            .for_each(|pheromone_value| assert_eq!(*pheromone_value, 0.0));

        pheromones.diffuse();

        // Assert all cells are zero after the diffuse
        pheromones
            .iter()
            .for_each(|pheromone_value| assert_eq!(*pheromone_value, 0.0));
    }

    #[test]
    fn diffuse_does_nothing_to_cell_when_pheromone_count_is_same_as_in_all_neighboring_cells() {
        let center = Point2::new(1.0, 1.0);
        let mut pheromones = Pheromones::new(3, 3, 1.0, false, None);

        // Assert all cells are 1.0 before the diffuse
        pheromones
            .iter()
            .for_each(|pheromone_value| assert_eq!(*pheromone_value, 1.0));

        pheromones.diffuse();

        // Assert center cell is 1.0 after the diffuse
        match pheromones.get_reading(center) {
            Some(reading) => assert_eq!(reading, 1.0),
            None => panic!("index ({}, {}) is out of range", center.x(), center.y()),
        }
    }

    // zelda has poor spatial reasoning
    // (0,0) (1,0) (2,0)
    // (0,1) (1,1) (2,1)
    // (0,2) (1,2) (2,2)

    #[test]
    fn neighbor_indexes_for_center_cell() {
        let expected_nw_index = (Some(0), Some(0));
        let expected_n_index = (Some(1), Some(0));
        let expected_ne_index = (Some(2), Some(0));
        let expected_e_index = (Some(2), Some(1));
        let expected_se_index = (Some(2), Some(2));
        let expected_s_index = (Some(1), Some(2));
        let expected_sw_index = (Some(0), Some(2));
        let expected_w_index = (Some(0), Some(1));
        #[rustfmt::skip]
        let [
            actual_nw_index,
            actual_n_index,
            actual_ne_index,
            actual_e_index,
            actual_se_index,
            actual_s_index,
            actual_sw_index,
            actual_w_index
        ] = neighbor_indexes(1, 1, 3, 3);

        assert_eq!(expected_nw_index, actual_nw_index);
        assert_eq!(expected_n_index, actual_n_index);
        assert_eq!(expected_ne_index, actual_ne_index);
        assert_eq!(expected_e_index, actual_e_index);
        assert_eq!(expected_se_index, actual_se_index);
        assert_eq!(expected_s_index, actual_s_index);
        assert_eq!(expected_sw_index, actual_sw_index);
        assert_eq!(expected_w_index, actual_w_index);
    }

    #[test]
    fn neighbor_indexes_for_corner_cell() {
        let expected_nw_index = (None, None);
        let expected_n_index = (Some(0), None);
        let expected_ne_index = (Some(1), None);
        let expected_e_index = (Some(1), Some(0));
        let expected_se_index = (Some(1), Some(1));
        let expected_s_index = (Some(0), Some(1));
        let expected_sw_index = (None, Some(1));
        let expected_w_index = (None, Some(0));
        #[rustfmt::skip]
        let [
            actual_nw_index,
            actual_n_index,
            actual_ne_index,
            actual_e_index,
            actual_se_index,
            actual_s_index,
            actual_sw_index,
            actual_w_index
        ] = neighbor_indexes(0, 0, 3, 3);

        assert_eq!(expected_nw_index, actual_nw_index);
        assert_eq!(expected_n_index, actual_n_index);
        assert_eq!(expected_ne_index, actual_ne_index);
        assert_eq!(expected_e_index, actual_e_index);
        assert_eq!(expected_se_index, actual_se_index);
        assert_eq!(expected_s_index, actual_s_index);
        assert_eq!(expected_sw_index, actual_sw_index);
        assert_eq!(expected_w_index, actual_w_index);
    }

    #[test]
    fn neighbor_indexes_for_edge_cell() {
        let expected_nw_index = (None, Some(0));
        let expected_n_index = (Some(0), Some(0));
        let expected_ne_index = (Some(1), Some(0));
        let expected_e_index = (Some(1), Some(1));
        let expected_se_index = (Some(1), Some(2));
        let expected_s_index = (Some(0), Some(2));
        let expected_sw_index = (None, Some(2));
        let expected_w_index = (None, Some(1));
        #[rustfmt::skip]
        let [
            actual_nw_index,
            actual_n_index,
            actual_ne_index,
            actual_e_index,
            actual_se_index,
            actual_s_index,
            actual_sw_index,
            actual_w_index
        ] = neighbor_indexes(0, 1, 3, 3);

        assert_eq!(expected_nw_index, actual_nw_index);
        assert_eq!(expected_n_index, actual_n_index);
        assert_eq!(expected_ne_index, actual_ne_index);
        assert_eq!(expected_e_index, actual_e_index);
        assert_eq!(expected_se_index, actual_se_index);
        assert_eq!(expected_s_index, actual_s_index);
        assert_eq!(expected_sw_index, actual_sw_index);
        assert_eq!(expected_w_index, actual_w_index);
    }
    #[test]
    fn diffuse_correctly_handles_corners() {
        let nw_corner = Point2::new(0.0, 0.0);
        let ne_corner = Point2::new(2.0, 0.0);
        let se_corner = Point2::new(2.0, 2.0);
        let sw_corner = Point2::new(0.0, 2.0);
        let expected_pheromone_level_after_diffuse = 4.0 / 9.0;
        let mut pheromones = Pheromones::new(3, 3, 1.0, false, None);

        // Assert all cells are 1.0 before the diffuse
        pheromones
            .iter()
            .for_each(|pheromone_value| assert_eq!(*pheromone_value, 1.0));

        pheromones.diffuse();

        // Assert edge cells are all equal after the diffuse
        match (
            pheromones.get_reading(nw_corner),
            pheromones.get_reading(ne_corner),
            pheromones.get_reading(se_corner),
            pheromones.get_reading(sw_corner),
        ) {
            (Some(n_reading), Some(e_reading), Some(s_reading), Some(w_reading)) => {
                assert_eq!(
                    (n_reading, e_reading, s_reading, w_reading),
                    (
                        expected_pheromone_level_after_diffuse,
                        expected_pheromone_level_after_diffuse,
                        expected_pheromone_level_after_diffuse,
                        expected_pheromone_level_after_diffuse,
                    )
                );
            }
            (None, _, _, _) => panic!(
                "nw index ({}, {}) is out of range",
                nw_corner.x(),
                nw_corner.y()
            ),
            (_, None, _, _) => panic!(
                "ne index ({}, {}) is out of range",
                ne_corner.x(),
                ne_corner.y()
            ),
            (_, _, None, _) => panic!(
                "se index ({}, {}) is out of range",
                se_corner.x(),
                se_corner.y()
            ),
            (_, _, _, None) => panic!(
                "sw index ({}, {}) is out of range",
                sw_corner.x(),
                sw_corner.y()
            ),
        }
    }

    #[test]
    fn diffuse_correctly_handles_edges() {
        let n_edge = Point2::new(1.0, 0.0);
        let e_edge = Point2::new(2.0, 1.0);
        let s_edge = Point2::new(1.0, 2.0);
        let w_edge = Point2::new(0.0, 1.0);
        let expected_pheromone_level_after_diffuse = 6.0 / 9.0;
        let mut pheromones = Pheromones::new(3, 3, 1.0, false, None);

        // Assert all cells are 1.0 before the diffuse
        pheromones
            .iter()
            .for_each(|pheromone_value| assert_eq!(*pheromone_value, 1.0));

        pheromones.diffuse();

        // Assert edge cells are all equal after the diffuse
        match (
            pheromones.get_reading(n_edge),
            pheromones.get_reading(e_edge),
            pheromones.get_reading(s_edge),
            pheromones.get_reading(w_edge),
        ) {
            (Some(n_reading), Some(e_reading), Some(s_reading), Some(w_reading)) => {
                assert_eq!(
                    (n_reading, e_reading, s_reading, w_reading),
                    (
                        expected_pheromone_level_after_diffuse,
                        expected_pheromone_level_after_diffuse,
                        expected_pheromone_level_after_diffuse,
                        expected_pheromone_level_after_diffuse
                    ),
                );
            }
            (None, _, _, _) => panic!("n index ({}, {}) is out of range", n_edge.x(), n_edge.y()),
            (_, None, _, _) => panic!("e index ({}, {}) is out of range", e_edge.x(), e_edge.y()),
            (_, _, None, _) => panic!("s index ({}, {}) is out of range", s_edge.x(), s_edge.y()),
            (_, _, _, None) => panic!("w index ({}, {}) is out of range", w_edge.x(), w_edge.y()),
        }
    }
}
