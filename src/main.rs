mod agent;
mod errors;
#[cfg(feature = "midi")]
mod midi;
mod pheromones;
mod point2;
mod swapper;
mod util;

pub use agent::Agent;
#[cfg(feature = "midi")]
use agent::AgentUpdate;
use circular_queue::CircularQueue;
use colorgrad::Gradient;
use colorgrad::{Color, CustomGradient};
use errors::SlimeError;
use log::{debug, error, info};
#[cfg(feature = "midi")]
use midi::{MidiInterface, MidiMessage};
use pheromones::Pheromones;
use pixels::{Pixels, SurfaceTexture};
pub use point2::Point2;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use std::time::Instant;
pub use swapper::Swapper;
use util::map_range;
use winit::{
    dpi::LogicalSize,
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

// General settings
pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;

// Agent settings
pub const AGENT_COUNT: usize = 100000;
pub const AGENT_COUNT_MAXIMUM: usize = 100000;
pub const AGENT_JITTER: f64 = 10.0;
pub const AGENT_SPEED_MIN: f64 = 0.5 + 10.0;
pub const AGENT_SPEED_MAX: f64 = 1.2 + 10.0;
pub const AGENT_TURN_SPEED: f64 = 12.0;
pub const AGENT_POSSIBLE_STARTING_HEADINGS: std::ops::Range<f64> = 0.0..360.0;
pub const DEPOSITION_AMOUNT: f64 = 1.0;

// Pheromone settings
/// Represents the rate at which pheromone signals disappear. A typical decay factor is 1/100 the rate of deposition
pub const DECAY_FACTOR: f64 = 0.01;

fn main() -> Result<(), SlimeError> {
    env_logger::init();

    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    debug!("opening {} by {} window", WIDTH, HEIGHT);

    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Slime Mold")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        debug!(
            "creating {} by {} surface_texture",
            window_size.width, window_size.height
        );
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    #[cfg(feature = "midi")]
    let mut midi_interface = MidiInterface::new(Some("Summit"))?;
    #[cfg(feature = "midi")]
    midi_interface.open()?;

    let mut world = World::new();

    let mut frame_time = 0.16;
    let mut time_of_last_frame_start = Instant::now();
    let mut frame_counter = 0;
    let mut fps_values = CircularQueue::with_capacity(5);
    let mut time_of_last_fps_counter_update = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        world.set_frame_time(frame_time);

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

            frame_time = time_of_last_frame_start.elapsed().as_secs_f64();
            time_of_last_frame_start = Instant::now();

            frame_counter += 1;

            if time_of_last_fps_counter_update.elapsed().as_secs() > 1 {
                time_of_last_fps_counter_update = Instant::now();
                let _ = fps_values.push(frame_counter);
                frame_counter = 0;

                let fps_sum: i32 = fps_values.iter().sum();
                let avg_fps = fps_sum as f64 / fps_values.len() as f64;
                info!("FPS {}", avg_fps.trunc());
            }
        }

        #[cfg(feature = "midi")]
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

            // Toggle B/W mode
            if input.key_pressed(VirtualKeyCode::B) {
                world.black_and_white_mode = !world.black_and_white_mode
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

struct World {
    agents: Vec<Agent>,
    frame_time: f64,
    gradient: Gradient,
    pheromones: Arc<RwLock<Pheromones>>,
    /// A toggle for rendering in color vs. black & white mode. Color mode has an FPS cost so we render in B&W by default
    black_and_white_mode: bool,
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    fn new() -> Self {
        info!("generating {} agents", AGENT_COUNT);
        info!(
            r#"
AGENT_COUNT_MAX	{:?}
AGENT_JITTER	{:?}
AGENT_SPEED_	{:?}
AGENT_SPEED_	{:?}
AGENT_TURN_SPEED	{:?}
AGENT_POSSIBLE_STRT_HDGS	{:?}
DEPOSITION_AMOUN	{:?}
DECAY_FACTOR	{:?}
"#,
            AGENT_COUNT_MAXIMUM,
            AGENT_JITTER,
            AGENT_SPEED_MIN,
            AGENT_SPEED_MAX,
            AGENT_TURN_SPEED,
            AGENT_POSSIBLE_STARTING_HEADINGS,
            DEPOSITION_AMOUNT,
            DECAY_FACTOR,
        );

        let agents: Vec<_> = (0..AGENT_COUNT)
            .into_iter()
            .map(|i| {
                let mut rng: StdRng = SeedableRng::from_entropy();
                let deposition_amount = if i == 0 {
                    DEPOSITION_AMOUNT * 20.0
                } else {
                    DEPOSITION_AMOUNT
                };
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
                    .rng(SeedableRng::from_entropy())
                    .build()
            })
            .collect();

        let pheromones = Arc::new(RwLock::new(Pheromones::new(
            WIDTH as usize,
            HEIGHT as usize,
            0.0,
            true,
            None, // Some(Box::new(generate_circular_static_gradient)),
        )));

        let gradient = CustomGradient::new()
            .colors(&[
                Color::from_rgb_u8(0, 0, 0),
                Color::from_rgb_u8(255, 251, 238),
            ])
            .build()
            .expect("failed to build gradient");

        Self {
            agents,
            frame_time: 0.0,
            gradient,
            pheromones,
            black_and_white_mode: true,
        }
    }

    fn set_frame_time(&mut self, frame_time: f64) {
        self.frame_time = frame_time;
    }

    /// Update the `World` internal state; bounce the box around the screen.
    fn update(&mut self) {
        let pheromones = &self.pheromones;
        let agents = &mut self.agents;
        // More stuff should be affected by delta_t in order to make the simulation run at the same speed
        // regardless of how fast the program is actually running. Right now it just affects agent speed.
        let delta_t = self.frame_time;

        agents.iter_mut().for_each(|agent| {
            let pheromones = pheromones
                .read()
                .expect("reading pheromones during agent update");
            let sensory_input = agent.sense(&pheromones);
            let rotation_towards_sensory_input = agent.judge_sensory_input(sensory_input);
            agent.rotate(rotation_towards_sensory_input);
            agent.move_in_direction_of_current_heading(delta_t);
        });

        let pheromones = &mut self.pheromones;
        let agents = &self.agents;

        pheromones
            .write()
            .expect("couldn't get mut ref to pheromones for deposit step")
            .deposit(agents);

        pheromones
            .write()
            .expect("couldn't get mut ref to pheromones for diffuse step")
            .diffuse();
        pheromones
            .write()
            .expect("couldn't get mut ref to pheromones for decay step")
            .decay();

        if self.agents.len() > AGENT_COUNT_MAXIMUM {
            self.agents.truncate(AGENT_COUNT_MAXIMUM)
        }
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    /// Assumes that pheromone grid and pixel FB have same dimensions
    fn draw(&self, frame: &mut [u8]) {
        let pixel_iter = frame.par_chunks_exact_mut(4);
        let gradient = &self.gradient;
        // TODO Grid doesn't support parallel iterators, what do?
        let pheromones: Vec<_> = self
            .pheromones
            .read()
            .expect("couldn't get lock on pheromones for draw")
            .iter()
            .map(ToOwned::to_owned)
            .collect();

        pixel_iter
            .zip_eq(pheromones.par_iter())
            .for_each(|(pixel, pheromone_value)| {
                // clamp to renderable range
                let pheromone_value = pheromone_value.clamp(0.0, 1.0);
                // map cell pheromone values to rgba pixels
                if self.black_and_white_mode == true {
                    let pheromone_value = map_range(pheromone_value, 0.0f64, 1.0f64, 0u8, 255u8);
                    pixel.copy_from_slice(&[
                        pheromone_value,
                        pheromone_value,
                        pheromone_value,
                        0xff,
                    ]);
                } else {
                    let (r, g, b, a) = gradient.at(pheromone_value).rgba_u8();
                    pixel.copy_from_slice(&[r, g, b, a]);
                }
            });
    }
}

// Notes on my synth are in the 36 to 96 range by default
// vel ranges from 0 to 127
#[cfg(feature = "midi")]
fn new_agent_from_midi(key: u8, vel: u8) -> Agent {
    let mut rng: StdRng = SeedableRng::from_entropy();
    let move_speed = rng.gen_range(AGENT_SPEED_MIN..AGENT_SPEED_MAX);
    let location = Point2::new(
        map_range(key as f64, 36.0, 96.0, 0.0, WIDTH as f64),
        map_range(vel as f64, 0.0, 127.0, 0.0, HEIGHT as f64),
    );

    let heading = rng.gen_range(AGENT_POSSIBLE_STARTING_HEADINGS);

    Agent::builder()
        .location(location)
        .heading(heading)
        .move_speed(move_speed)
        .jitter(AGENT_JITTER)
        .deposition_amount(DEPOSITION_AMOUNT)
        .rotation_speed(AGENT_TURN_SPEED)
        .rng(rng)
        .build()
}

#[cfg(feature = "midi")]
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
