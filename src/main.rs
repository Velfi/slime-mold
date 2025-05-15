mod agent;
mod errors;
#[cfg(feature = "midi")]
mod midi;
mod pheromones;
mod point2;
mod rect;
mod settings;
mod swapper;
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
use notify::{Error as NotifyError, Event as NotifyEvent, RecursiveMode, Watcher};
pub use point2::Point2;
use settings::Settings;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;
pub use swapper::Swapper;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
    keyboard::KeyCode,
    window::Window,
};
use winit_input_helper::WinitInputHelper;
use world::World;

pub const DEFAULT_SETTINGS_FILE: &str = "simulation_settings.toml";

#[cfg(target_os = "macos")]
pub type SettingsWatcherComponents = (Receiver<Result<NotifyEvent, NotifyError>>, FsEventWatcher);
#[cfg(target_os = "linux")]
pub type SettingsWatcherComponents = (Receiver<Result<NotifyEvent, NotifyError>>, INotifyWatcher);
#[cfg(target_os = "windows")]
pub type SettingsWatcherComponents = (
    Receiver<Result<NotifyEvent, NotifyError>>,
    ReadDirectoryChangesWatcher,
);

async fn run(event_loop: EventLoop<()>, window: Arc<Window>, initial_settings: Settings) {
    let current_settings = initial_settings.clone(); // Used for config width/height initially
    let (settings_update_receiver, _settings_update_watcher) =
        setup_settings_file_watcher(DEFAULT_SETTINGS_FILE)
            .expect("Failed to setup settings file watcher");

    let mut input = WinitInputHelper::new();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    // The surface needs to live as long as the window that created it.
    // Arc::clone(&window) ensures the surface holds a reference to the window.
    let surface = instance
        .create_surface(Arc::clone(&window))
        .expect("Failed to create surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None, // Trace path
        )
        .await
        .unwrap();

    // Wrap device and queue in Arc for sharing
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb()) // Prefer sRGB
        .unwrap_or(surface_caps.formats[0]);

    let present_mode = surface_caps
        .present_modes
        .iter()
        .copied()
        .find(|&mode| mode == wgpu::PresentMode::Mailbox)
        .unwrap_or_else(|| {
            surface_caps
                .present_modes
                .iter()
                .copied()
                .find(|&mode| mode == wgpu::PresentMode::Immediate)
                .unwrap_or(wgpu::PresentMode::Fifo) // Fallback to Fifo
        });
    info!("Selected PresentMode: {:?}", present_mode);

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: current_settings.window_width,
        height: current_settings.window_height,
        present_mode, // Use the selected present mode
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 1,
    };
    surface.configure(&device, &config);

    // Sampler remains, as it's for sampling the world's display texture
    let sim_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Simulation Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Shader
    let shader_source = wgpu::ShaderSource::Wgsl(
        fs::read_to_string("src/shaders/display.wgsl")
            .expect("Should have been able to read the file")
            .into(),
    );

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader Module"),
        source: shader_source,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Texture Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let mut world = World::new(current_settings, Arc::clone(&device), Arc::clone(&queue));

    // Initialize bind_group using the texture view from the World
    let mut bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Texture Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                // Get the display texture view from the world instance
                resource: wgpu::BindingResource::TextureView(world.get_display_texture_view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sim_sampler),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "vs_main",
            buffers: &[], // No vertex buffers, vertices generated in shader
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(), // Simplified
        multiview: None,
    });

    #[cfg(feature = "midi")]
    let mut midi_interface =
        MidiInterface::new(Some("Summit")).expect("Midi interface creation failed");
    #[cfg(feature = "midi")]
    midi_interface.open().expect("Midi interface open failed");

    let _last_log_time = Instant::now();
    let _frame_times: CircularQueue<f32> = CircularQueue::with_capacity(60);

    let mut frame_time = 0.16;
    let mut time_of_last_frame_start = Instant::now();
    let mut frame_counter = 0;
    let mut fps_values = CircularQueue::with_capacity(5);
    let mut time_of_last_fps_counter_update = Instant::now();

    let window_for_event_loop = Arc::clone(&window); // Clone Arc for the event loop closure

    event_loop.set_control_flow(ControlFlow::Poll);
    let result = event_loop.run(move |event, elwt: &EventLoopWindowTarget<()>| {
        // Process events with WinitInputHelper first
        if input.update(&event) {
            // Input helper consumed the event, check its state
            if input.key_pressed(KeyCode::Escape) || input.close_requested() || input.destroyed() {
                elwt.exit();
                return; // Exit event processing for this iteration
            }
            if input.key_pressed(KeyCode::KeyD) {
                world.toggle_dynamic_gradient();
            }
            if let Some(size) = input.window_resized() {
                info!("Window resized (via input helper) to: {:?}", size);
                if size.width > 0 && size.height > 0 {
                    config.width = size.width;
                    config.height = size.height;
                    surface.configure(&device, &config);

                    // World handles its own texture resizing internally now
                    world.resize(config.width, config.height);

                    // Recreate bind group with the new texture view from the resized world
                    bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Texture Bind Group (Resized)"),
                        layout: &bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    world.get_display_texture_view(),
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&sim_sampler),
                            },
                        ],
                    });
                }
            }
        }

        // After input helper, handle specific winit events
        match event {
            Event::WindowEvent {
                event: window_event,
                window_id,
            } if window_id == window_for_event_loop.id() => match window_event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => {
                    // This might be redundant if input.window_resized() already handled it,
                    // but direct handling ensures wgpu surface is configured.
                    info!("Window resized (direct event) to: {:?}", physical_size);
                    if physical_size.width > 0 && physical_size.height > 0 {
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        surface.configure(&device, &config);

                        // World handles its own texture resizing internally now
                        world.resize(config.width, config.height);

                        // Recreate bind group with the new texture view from the resized world
                        bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Texture Bind Group (Resized)"),
                            layout: &bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        world.get_display_texture_view(),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&sim_sampler),
                                },
                            ],
                        });
                    }
                }
                WindowEvent::RedrawRequested => {
                    world.set_frame_time(frame_time); // Set frame time before simulation step (world.update() will use this)

                    // World's internal display texture is updated during world.update()
                    // No direct world.draw() call or frame_data needed here.

                    match surface.get_current_texture() {
                        Ok(output_frame) => {
                            let view = output_frame
                                .texture
                                .create_view(&wgpu::TextureViewDescriptor::default());
                            let mut encoder =
                                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("Render Encoder"),
                                });
                            {
                                let mut render_pass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: Some("Render Pass"),
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                                        r: 0.0,
                                                        g: 0.0,
                                                        b: 0.0,
                                                        a: 1.0,
                                                    }),
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        depth_stencil_attachment: None,
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                    });
                                render_pass.set_pipeline(&render_pipeline);
                                render_pass.set_bind_group(0, &bind_group, &[]);
                                render_pass.draw(0..4, 0..1);
                            }
                            queue.submit(std::iter::once(encoder.finish()));
                            output_frame.present();
                        }
                        Err(wgpu::SurfaceError::Lost) => {
                            config.width = world.window_width();
                            config.height = world.window_height();
                            if config.width > 0 && config.height > 0 {
                                surface.configure(&device, &config);
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            error!("SurfaceError::OutOfMemory");
                            elwt.exit();
                        }
                        Err(e) => {
                            error!("Unhandled surface error: {:?}, reconfiguring", e);
                            config.width = world.window_width();
                            config.height = world.window_height();
                            if config.width > 0 && config.height > 0 {
                                surface.configure(&device, &config);
                            }
                        }
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
                        window_for_event_loop
                            .set_title(&format!("Slime Mold - FPS: {:.0}", avg_fps.trunc()));
                    }
                }
                _ => {} // Other window events handled by input.update or ignored
            },
            Event::AboutToWait => {
                world.update();
                window_for_event_loop.request_redraw();
            }
            _ => {} // Other event types
        }

        // Handle MIDI events if feature is enabled
        #[cfg(feature = "midi")]
        for midi_event in midi_interface.pending_events() {
            match midi_event {
                MidiMessage::NoteOn { key, vel } => {
                    let (key_val, vel_val) = (key.as_int(), vel.as_int());
                    info!("hit note {} with vel {}", key_val, vel_val);
                    let mut new_agents: Vec<_> = (0..5)
                        .into_iter()
                        .map(|_| new_agent_from_midi(key_val, vel_val))
                        .collect();
                    world.agents.append(&mut new_agents); // Assuming world.agents is public or has an add_agents method
                }
                MidiMessage::NoteOff { key, vel } => {
                    trace!("released note {} with vel {}", key.as_int(), vel.as_int());
                }
                MidiMessage::Aftertouch { key, vel } => {
                    trace!("Aftertouch: key {} vel {}", key.as_int(), vel.as_int());
                }
                MidiMessage::Controller { controller, value } => {
                    trace!(
                        "Controller {} value {}",
                        controller.as_int(),
                        value.as_int()
                    );
                    handle_controller_input(&mut world, controller.as_int(), value.as_int());
                }
                _ => (),
            }
        }

        // Handle settings file updates
        match settings_update_receiver.try_recv() {
            Ok(_event) => {
                info!("Settings file has changed. Reloading it and updating the World");
                if let Err(e) = world.reload_settings() {
                    error!("failed to reload settings file: {}", e)
                }
            }
            Err(e) => {
                if e != std::sync::mpsc::TryRecvError::Empty {
                    // Don't trace if just empty
                    trace!("settings_update_receiver watch error: {:?}", e);
                }
            }
        }
    });

    if let Err(e) = result {
        error!("Event loop error: {:?}", e);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenv::dotenv();
    env_logger::init();
    let initial_settings = Settings::load_from_file(DEFAULT_SETTINGS_FILE)?;

    let event_loop = EventLoop::new().unwrap();

    let window = Arc::new(
        winit::window::WindowBuilder::new()
            .with_title("Slime Mold")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                initial_settings.window_width,
                initial_settings.window_height,
            ))
            .build(&event_loop)?,
    );

    pollster::block_on(run(event_loop, window, initial_settings));

    Ok(())
}

#[cfg(target_os = "windows")]
fn setup_settings_file_watcher(
    path: impl AsRef<Path>,
) -> Result<SettingsWatcherComponents, Box<dyn std::error::Error>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher: ReadDirectoryChangesWatcher = Watcher::new(tx, notify::Config::default())?;
    watcher.watch(path.as_ref(), RecursiveMode::Recursive)?;
    Ok((rx, watcher))
}

#[cfg(target_os = "macos")]
fn setup_settings_file_watcher(
    path: impl AsRef<Path>,
) -> Result<SettingsWatcherComponents, Box<dyn std::error::Error>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher: FsEventWatcher =
        notify::recommended_watcher(move |res: Result<NotifyEvent, NotifyError>| {
            if let Err(e) = tx.send(res) {
                error!("Error sending file watch event: {}", e);
            }
        })?;
    watcher.watch(path.as_ref(), RecursiveMode::Recursive)?;
    Ok((rx, watcher))
}

#[cfg(target_os = "linux")]
fn setup_settings_file_watcher(
    path: impl AsRef<Path>,
) -> Result<SettingsWatcherComponents, Box<dyn std::error::Error>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut watcher: INotifyWatcher =
        notify::recommended_watcher(move |res: Result<NotifyEvent, NotifyError>| {
            if let Err(e) = tx.send(res) {
                error!("Error sending file watch event: {}", e);
            }
        })?;
    watcher.watch(path.as_ref(), RecursiveMode::Recursive)?;
    Ok((rx, watcher))
}

// Helper for MIDI if needed, placeholder
#[cfg(feature = "midi")]
fn new_agent_from_midi(_key: u8, _vel: u8) -> Agent {
    // Placeholder implementation - you'll need to define this based on your Agent::new or similar
    // This function was referenced but not defined in the provided main.rs snippet when MIDI was active
    let temp_settings = Settings::default(); // Or fetch current world settings
    Agent::new_from_settings(&temp_settings)
}

#[cfg(feature = "midi")]
fn handle_controller_input(_world: &mut World, _controller: u8, _value: u8) {
    // Placeholder
}
