mod agent;
mod pheromones;
mod point2;
mod swapper;
mod util;

pub use agent::Agent;
use log::error;
use pheromones::Pheromones;
use pixels::{Error, Pixels, SurfaceTexture};
pub use point2::Point2;
use rand::prelude::*;
use std::{cell::RefCell, rc::Rc};
pub use swapper::Swapper;
use util::map_range;
use winit::{
    dpi::LogicalSize,
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

pub const WIDTH: u32 = 800;
pub const HEIGHT: u32 = 600;

struct World {
    agents: Vec<Agent>,
    pheromones: Rc<RefCell<Pheromones>>,
    rng: Rc<RefCell<ThreadRng>>,
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Hello Pixels")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };
    let mut world = World::new();

    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
            world.draw(pixels.get_frame());
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                pixels.resize(size.width, size.height);
            }

            // Update internal state and request a redraw
            world.update();
            window.request_redraw();
        }
    });
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    fn new() -> Self {
        let rng = Rc::new(RefCell::new(thread_rng()));

        let agents: Vec<_> = (0..1000)
            .into_iter()
            .map(|_| {
                let rng = rng.as_ref();
                let rng = &mut *rng.borrow_mut();

                let location = Point2::new(
                    rng.gen_range(0.0..(WIDTH as f64)),
                    rng.gen_range(0.0..(HEIGHT as f64)),
                );
                let heading = rng.gen_range(0.0..360.0) as f64;
                Agent::builder()
                    .location(location)
                    .heading(heading)
                    .move_speed(0.0)
                    .build()
            })
            .collect();

        let pheromones = Rc::new(RefCell::new(Pheromones::new(
            WIDTH as usize,
            HEIGHT as usize,
            0.0,
        )));

        Self {
            agents,
            pheromones,
            rng,
        }
    }

    /// Update the `World` internal state; bounce the box around the screen.
    fn update(&mut self) {
        let pheromones = self.pheromones.as_ref();
        let rng = self.rng.as_ref();
        self.agents.iter_mut().for_each(|agent| {
            let rng = &mut *rng.borrow_mut();
            let sensory_input = agent.sense(&pheromones.borrow());
            let rotation_towards_sensory_input = agent.judge_sensory_input(sensory_input, rng);
            agent.rotate(rotation_towards_sensory_input);
            agent.move_in_direction_of_current_heading();
            (&mut pheromones.borrow_mut()).deposit(&agent);
        });

        pheromones.borrow_mut().diffuse();
        pheromones.borrow_mut().decay();
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    /// Assumes that pheromone grid and pixel FB have same dimensions
    fn draw(&self, frame: &mut [u8]) {
        let pixel_iter = frame.chunks_exact_mut(4);
        let pheromone_iter_len = self.pheromones.borrow().len();
        assert!(
            pixel_iter.len() == pheromone_iter_len,
            "Pixel FB {{len = {}}} and pheromone grid {{len = {}}} length mismatch",
            pixel_iter.len(),
            pheromone_iter_len
        );

        for (pixel, pheromone_value) in pixel_iter.zip(self.pheromones.borrow().iter()) {
            // clamp to renderable range
            let pheromone_value = pheromone_value.clamp(0.0, 1.0);
            // map from float to u8
            let pheromone_value = map_range(pheromone_value, 0.0f64, 1.0f64, 0u8, 255u8);
            let rgba = [pheromone_value, pheromone_value, pheromone_value, 0xff];

            pixel.copy_from_slice(&rgba);
        }
    }
}
