#[derive(Clone, Copy, Debug)]
#[derive(Default)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl std::fmt::Display for Point2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(x: {}, y: {})", self.x, self.y)
    }
}

impl Point2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn move_absolute(&mut self, x: f32, y: f32) {
        self.x = x;
        self.y = y;
    }
}

impl Point2 {
    pub fn distance_to(&self, other: &Point2) -> f32 {
        ((other.x - self.x).powi(2) + (other.y - self.y).powi(2)).sqrt()
    }
}

impl Point2 {
    pub fn clamp(&mut self, x_min: f32, y_min: f32, x_max: f32, y_max: f32) {
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

impl PartialEq for Point2 {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl Point2 {
    pub fn move_relative(&mut self, x: f32, y: f32) {
        self.x += x;
        self.y += y;
    }
}

