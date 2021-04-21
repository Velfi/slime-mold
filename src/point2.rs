use num::Float;
use std::ops::AddAssign;

#[derive(Clone, Copy, Debug)]
pub struct Point2<T> {
    x: T,
    y: T,
}

impl<T: Copy> Point2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    pub fn x(&self) -> T {
        self.x
    }

    pub fn y(&self) -> T {
        self.y
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
        ((other.x() - self.x()).powi(2) + (other.y() - self.y()).powi(2)).sqrt()
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

impl Into<Point2<usize>> for Point2<f64> {
    fn into(self) -> Point2<usize> {
        Point2 {
            x: self.x.round() as usize,
            y: self.y.round() as usize,
        }
    }
}
