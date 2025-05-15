use crate::Point2;
use std::ops::Add;

/// A Rectangle defined by its top left corner, width and height.
#[derive(Copy, Clone, Debug)]
pub struct Rect<T> {
    /// The x coordinate of the top left corner.
    pub x: T,
    /// The y coordinate of the top left corner.
    pub y: T,
    /// The rectangle's width.
    pub width: T,
    /// The rectangle's height.
    pub height: T,
}

impl<T: Copy> Rect<T> {
    pub fn new(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl<T> Rect<T>
where
    T: Add<Output = T> + Copy,
{
    pub fn x_min(&self) -> T {
        self.x
    }

    pub fn x_max(&self) -> T {
        self.x + self.width
    }

    pub fn y_min(&self) -> T {
        self.y
    }

    pub fn y_max(&self) -> T {
        self.y + self.height
    }
}

impl Rect<f32> {
    pub fn contains(&self, other: &Point2) -> bool {
        other.x >= self.x_min()
            && other.x < self.x_max()
            && other.y >= self.y_min()
            && other.y < self.y_max()
    }
}

impl Rect<u32> {
    pub fn clamp(&self, other: &mut Point2) {
        if other.x >= self.x_min() as f32
            && other.x < self.x_max() as f32
            && other.y >= self.y_min() as f32
            && other.y < self.y_max() as f32
        {
        } else {
            let x = other
                .x
                .clamp(self.x_min() as f32, self.x_max() as f32 - 1.0);
            let y = other
                .y
                .clamp(self.y_min() as f32, self.y_max() as f32 - 1.0);

            other.move_absolute(x, y)
        }
    }
}
