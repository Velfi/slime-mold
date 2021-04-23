mod agent;
mod errors;
mod midi;
mod pheromones;
mod point2;
mod swapper;
mod util;

pub use agent::Agent;
use agent::AgentUpdate;
use colorgrad::Gradient;
use errors::SlimeError;
use log::{debug, error, info};
use midi::MidiInterface;
use midly::MidiMessage;
use pheromones::Pheromones;
use pixels::{Pixels, SurfaceTexture};
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

pub const WIDTH: u32 = 1000;
pub const HEIGHT: u32 = 1000;
pub const AGENT_COUNT: usize = 0;
pub const AGENT_COUNT_MAXIMUM: usize = 100000;
pub const AGENT_JITTER: f64 = 10.0;
pub const AGENT_SPEED_MIN: f64 = 0.5;
pub const AGENT_SPEED_MAX: f64 = 1.2;
pub const AGENT_TURN_SPEED: f64 = 12.0;
pub const AGENT_POSSIBLE_STARTING_HEADINGS: std::ops::Range<f64> = 0.0..360.0;
pub const DEPOSITION_AMOUNT: f64 = 1.0;
pub const DECAY_FACTOR: f64 = DEPOSITION_AMOUNT / 100.0;

struct World {
    agents: Vec<Agent>,
    pheromones: Rc<RefCell<Pheromones>>,
    rng: Rc<RefCell<ThreadRng>>,
    _gradient: Gradient,
}

fn main() -> Result<(), SlimeError> {
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
    let mut midi_interface = MidiInterface::new(Some("Summit"))?;
    midi_interface.open()?;

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

        for event in midi_interface.pending_events() {
            match event {
                MidiMessage::NoteOn { key, vel } => {
                    let (key, vel) = (key.as_int(), vel.as_int());
                    info!("hit note {} with vel {}", key, vel);

                    let mut new_agents: Vec<_> = (0..5)
                        .into_iter()
                        .map(|_| new_agent_from_midi(key, vel))
                        .collect();

                    world.agents.append(&mut new_agents)
                }
                MidiMessage::NoteOff { key, vel } => {
                    debug!("released note {} with vel {}", key, vel);
                }
                MidiMessage::Aftertouch { key, vel } => {
                    debug!("Aftertouch: key {} vel {}", key, vel)
                }
                MidiMessage::Controller { controller, value } => {
                    debug!("Controller {} value {}", controller, value);
                    handle_controller_input(&mut world, controller.as_int(), value.as_int());
                }
                _ => (),
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

        let agents: Vec<_> = (0..AGENT_COUNT)
            .into_iter()
            .map(|i| {
                let deposition_amount = if i == 0 {
                    DEPOSITION_AMOUNT * 20.0
                } else {
                    DEPOSITION_AMOUNT
                };

                let rng = rng.as_ref();
                let rng = &mut *rng.borrow_mut();
                let move_speed = rng.gen_range(AGENT_SPEED_MIN..AGENT_SPEED_MAX);

                let location = Point2::new(
                    rng.gen_range(0.0..(WIDTH as f64)),
                    rng.gen_range(0.0..(HEIGHT as f64)),
                );
                let heading = rng.gen_range(AGENT_POSSIBLE_STARTING_HEADINGS);
                Agent::builder()
                    .location(location)
                    .heading(heading)
                    .move_speed(move_speed)
                    .jitter(AGENT_JITTER)
                    .deposition_amount(deposition_amount)
                    .rotation_speed(AGENT_TURN_SPEED)
                    .build()
            })
            .collect();

        let pheromones = Rc::new(RefCell::new(Pheromones::new(
            WIDTH as usize,
            HEIGHT as usize,
            0.0,
            true,
            None, // Some(Box::new(generate_circular_static_gradient)),
        )));

        let _gradient = colorgrad::turbo();

        Self {
            agents,
            _gradient,
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

        if self.agents.len() > AGENT_COUNT_MAXIMUM {
            self.agents.truncate(AGENT_COUNT_MAXIMUM)
        }
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
            pixel.copy_from_slice(&[pheromone_value, pheromone_value, pheromone_value, 0xff]);

            // let (r, g, b, a) = self._gradient.at(pheromone_value).rgba_u8();
            // pixel.copy_from_slice(&[r, g, b, a]);
        }
    }
}

// Notes on my synth are in the 36 to 96 range by default
// vel ranges from 0 to 127
fn new_agent_from_midi(key: u8, vel: u8) -> Agent {
    let rng = fastrand::Rng::new();
    let move_speed = map_range(rng.f64(), 0.0, 1.0, AGENT_SPEED_MIN, AGENT_SPEED_MAX);
    let location = Point2::new(
        map_range(key as f64, 36.0, 96.0, 0.0, WIDTH as f64),
        map_range(vel as f64, 0.0, 127.0, 0.0, HEIGHT as f64),
    );

    let heading = map_range(
        rng.f64(),
        0.0,
        1.0,
        AGENT_POSSIBLE_STARTING_HEADINGS.start,
        AGENT_POSSIBLE_STARTING_HEADINGS.end,
    );

    Agent::builder()
        .location(location)
        .heading(heading)
        .move_speed(move_speed)
        .jitter(AGENT_JITTER)
        .deposition_amount(DEPOSITION_AMOUNT)
        .rotation_speed(AGENT_TURN_SPEED)
        .build()
}

fn handle_controller_input(world: &mut World, controller: u8, value: u8) {
    match controller {
        // Noise Level
        27 => {
            let jitter = Some(map_range(value as f64, 0.0, 127.0, 2.0, 40.0));
            let agent_update = AgentUpdate {
                jitter,
                ..Default::default()
            };

            world
                .agents
                .iter_mut()
                .for_each(|agent| agent.apply_update(&agent_update));
        }
        // Filter Freq
        29 => {
            let deposition_amount = Some(map_range(value as f64, 0.0, 127.0, 0.2, 4.0));
            let agent_update = AgentUpdate {
                deposition_amount,
                ..Default::default()
            };

            world
                .agents
                .iter_mut()
                .for_each(|agent| agent.apply_update(&agent_update));
        }
        _ => (),
    };
}
