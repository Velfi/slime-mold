mod agent;
mod errors;
#[cfg(feature = "midi")]
mod midi;
mod pheromones;
mod point2;
mod rect;
mod settings;
mod swapper;
mod util;
mod world;

pub use agent::Agent;
use circular_queue::CircularQueue;
use log::{error, info, trace};
#[cfg(feature = "midi")]
use midi::{MidiInterface, MidiMessage};
#[cfg(target_os = "macos")]
use notify::FsEventWatcher;
#[cfg(target_os = "linux")]
use notify::INotifyWatcher;
#[cfg(target_os = "windows")]
use notify::ReadDirectoryChangesWatcher;
use notify::{RecursiveMode, Watcher};
use pheromones::Pheromones;
use pixels::{Pixels, SurfaceTexture};
pub use point2::Point2;
use settings::Settings;
use std::path::Path;
use std::sync::mpsc::Receiver;
use std::time::Instant;
pub use swapper::Swapper;
use winit::{event::Event, event_loop::EventLoop, keyboard::KeyCode};
use winit_input_helper::WinitInputHelper;
use world::World;

pub const DEFAULT_SETTINGS_FILE: &str = "simulation_settings.toml";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenv::dotenv();
    env_logger::init();
    let initial_settings = Settings::load_from_file(DEFAULT_SETTINGS_FILE)?;
    let (settings_update_receiver, _settings_update_watcher) =
        setup_settings_file_watcher(DEFAULT_SETTINGS_FILE);

    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();

    let window = winit::window::WindowBuilder::new()
        .with_title("Slime Mold")
        .with_inner_size(winit::dpi::PhysicalSize::new(
            initial_settings.window_width,
            initial_settings.window_height,
        ))
        .build(&event_loop)?;
    let mut pixels = Pixels::new(
        initial_settings.window_width,
        initial_settings.window_height,
        SurfaceTexture::new(
            initial_settings.window_width,
            initial_settings.window_height,
            &window,
        ),
    )?;

    #[cfg(feature = "midi")]
    let mut midi_interface = MidiInterface::new(Some("Summit"))?;
    #[cfg(feature = "midi")]
    midi_interface.open()?;

    let mut world = World::new(initial_settings);

    let mut frame_time = 0.16;
    let mut time_of_last_frame_start = Instant::now();
    let mut frame_counter = 0;
    let mut fps_values = CircularQueue::with_capacity(5);
    let mut time_of_last_fps_counter_update = Instant::now();

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run(|event, _event_loop_window_target| {
        world.set_frame_time(frame_time);

        if let Event::WindowEvent {
            event: winit::event::WindowEvent::RedrawRequested,
            window_id,
        } = event
        {
            if window_id == window.id() {
                world.draw(pixels.frame_mut());
                if pixels
                    .render()
                    .map_err(|e| error!("pixels.render() failed: {}", e))
                    .is_err()
                {
                    std::process::exit(0);
                }

                frame_time = time_of_last_frame_start.elapsed().as_secs_f32();
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

        if input.update(&event) {
            if input.key_pressed(KeyCode::Escape) || input.close_requested() || input.destroyed() {
                std::process::exit(0);
            }

            if input.key_pressed(KeyCode::KeyB) {
                world.toggle_black_and_white_mode();
            }

            if input.key_pressed(KeyCode::KeyD) {
                world.toggle_dynamic_gradient();
            }

            if let Some(size) = input.window_resized() {
                pixels
                    .resize_surface(size.width, size.height)
                    .expect("couldn't resize surface");
            }

            world.update();
            window.request_redraw();
        }

        match settings_update_receiver.try_recv() {
            Ok(_event) => {
                info!("Settings file has changed. Reloading it and updating the World");
                if let Err(e) = world.reload_settings() {
                    error!("failed to reload settings file: {}", e)
                }
            }
            Err(e) => {
                trace!("settings_update_receiver watch error: {:?}", e);
            }
        }
    })?;

    Ok(())
}

#[cfg(target_os = "windows")]
fn setup_settings_file_watcher(
    path: impl AsRef<Path>,
) -> (
    Receiver<notify::Result<notify::Event>>,
    ReadDirectoryChangesWatcher,
) {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |res| tx.send(res).unwrap())
        .expect("couldn't create file change watcher");

    watcher
        .watch(path, RecursiveMode::Recursive)
        .expect("couldn't start file change watcher");

    (rx, watcher)
}

#[cfg(target_os = "macos")]
fn setup_settings_file_watcher(
    path: impl AsRef<Path>,
) -> (Receiver<notify::Result<notify::Event>>, FsEventWatcher) {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher: FsEventWatcher = notify::recommended_watcher(move |res| tx.send(res).unwrap())
        .expect("couldn't create file change watcher");

    watcher
        .watch(path.as_ref(), RecursiveMode::Recursive)
        .expect("couldn't start file change watcher");

    (rx, watcher)
}

#[cfg(target_os = "linux")]
fn setup_settings_file_watcher(
    path: impl AsRef<Path>,
) -> (Receiver<notify::Result<notify::Event>>, INotifyWatcher) {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher: RecommendedWatcher = Watcher::new_immediate(move |res| tx.send(res).unwrap())
        .expect("couldn't create file change watcher");

    watcher
        .watch(path, RecursiveMode::Recursive)
        .expect("couldn't start file change watcher");

    (rx, watcher)
}
