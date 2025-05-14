use num::Float;
use std::{fmt::Display, ops::AddAssign};

#[derive(Clone, Copy, Debug)]
pub struct Point2<T> {
    pub x: T,
    pub y: T,
}

impl<T: Display> Display for Point2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(x: {}, y: {})", self.x, self.y)
    }
}

impl<T: Copy> Point2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    pub fn move_absolute(&mut self, x: T, y: T) {
        self.x = x;
        self.y = y;
    }
}

// TODO make generic across float
impl<T> Point2<T>
where
    T: Float,
{
    pub fn distance_to(&self, other: &Point2<T>) -> T {
        ((other.x - self.x).powi(2) + (other.y - self.y).powi(2)).sqrt()
    }
}

impl<T> Point2<T>
where
    T: PartialOrd,
{
    pub fn clamp(&mut self, x_min: T, y_min: T, x_max: T, y_max: T) {
        if self.x < x_min {
            self.x = x_min
        } else if self.x > x_max {
            self.x = x_max
        }

        if self.y < y_min {
            self.y = y_min
        } else if self.y > y_max {
            self.y = y_max
        }
    }
}

impl<T> PartialEq for Point2<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl<T> Point2<T>
where
    T: AddAssign,
{
    pub fn move_relative(&mut self, x: T, y: T) {
        self.x += x;
        self.y += y;
    }
}

impl<T: Default> Default for Point2<T> {
    fn default() -> Self {
        Point2 {
            x: T::default(),
            y: T::default(),
        }
    }
}

impl From<Point2<f64>> for Point2<usize> {
    fn from(val: Point2<f64>) -> Self {
        Point2 {
            x: val.x.round() as usize,
            y: val.y.round() as usize,
        }
    }
}

impl From<Point2<f64>> for Point2<u32> {
    fn from(val: Point2<f64>) -> Self {
        Point2 {
            x: val.x.round().clamp(u32::MIN as f64, u32::MAX as f64) as u32,
            y: val.y.round().clamp(u32::MIN as f64, u32::MAX as f64) as u32,
        }
    }
}
