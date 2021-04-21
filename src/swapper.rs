use std::mem;

/// Swapper is a wrapper for two things of the same type. It's meant to be
/// useful in situations where you need to iterate over one collection and
/// write the results to a second and then write new data to the first
/// collection by iterating over the second collection.
pub struct Swapper<T> {
    a: T,
    b: T,
}

impl<T> Swapper<T> {
    pub fn new(a: T, b: T) -> Self {
        Self { a, b }
    }

    pub fn swap(&mut self) {
        mem::swap(&mut self.a, &mut self.b)
    }

    pub fn a(&self) -> &T {
        &self.a
    }

    pub fn b(&self) -> &T {
        &self.b
    }

    pub fn mut_a(&mut self) -> &mut T {
        &mut self.a
    }

    pub fn mut_b(&mut self) -> &mut T {
        &mut self.b
    }

    pub fn read_a_write_b(&mut self) -> (&T, &mut T) {
        (&self.a, &mut self.b)
    }

    pub fn write_a_read_b(&mut self) -> (&mut T, &T) {
        (&mut self.a, &self.b)
    }
}
