use bytemuck::cast_slice_mut;
use log::info;
use num_format::{Locale, ToFormattedString};
use slime_mold::lut_manager::LutManager;
use slime_mold::presets::init_preset_manager;
use slime_mold::settings::Settings;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use wgpu::{Backends, Buffer, BufferUsages, Device, Instance, Queue, TextureUsages};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::KeyCode,
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;

mod bind_group_manager;
mod pipeline_manager;
mod shader_manager;
mod text_renderer;

use bind_group_manager::BindGroupManager;
use pipeline_manager::PipelineManager;
use shader_manager::ShaderManager;
use text_renderer::TextRenderer;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimSizeUniform {
    width: u32,
    height: u32,
    decay_factor: f32,
    agent_jitter: f32,
    agent_speed_min: f32,
    agent_speed_max: f32,
    agent_turn_speed: f32,
    agent_sensor_angle: f32,
    agent_sensor_distance: f32,
    diffusion_rate: f32,
    pheromone_deposition_amount: f32,
    blur_radius: f32,
    blur_sigma: f32,
    _pad: [u32; 1],
}

impl SimSizeUniform {
    fn new(width: u32, height: u32, decay_factor: f32, settings: &Settings) -> Self {
        Self {
            width,
            height,
            decay_factor,
            agent_jitter: settings.agent_jitter,
            agent_speed_min: settings.agent_speed_min,
            agent_speed_max: settings.agent_speed_max,
            agent_turn_speed: settings.agent_turn_speed,
            agent_sensor_angle: settings.agent_sensor_angle,
            agent_sensor_distance: settings.agent_sensor_distance,
            diffusion_rate: settings.pheromone_diffusion_rate,
            pheromone_deposition_amount: settings.pheromone_deposition_amount,
            blur_radius: settings.blur_radius,
            blur_sigma: settings.blur_sigma,
            _pad: [0],
        }
    }
}

fn format_float_dynamic(val: f32) -> String {
    let s = format!("{}", val);
    if s.contains('.') {
        let s = s.trim_end_matches('0').trim_end_matches('.');
        if s.is_empty() {
            "0.0".to_string()
        } else {
            s.to_string()
        }
    } else {
        format!("{}.0", s)
    }
}

fn update_settings(
    settings: &mut Settings,
    current_preset_name: &mut String,
    sim_size_buffer: &Buffer,
    queue: &Queue,
    physical_width: u32,
    physical_height: u32,
) {
    *current_preset_name = "CUSTOM".to_string();
    let sim_size_uniform = SimSizeUniform::new(
        physical_width,
        physical_height,
        settings.pheromone_decay_factor,
        settings,
    );
    queue.write_buffer(sim_size_buffer, 0, bytemuck::bytes_of(&sim_size_uniform));
}

fn reassign_agent_speeds(
    agent_buffer: &Buffer,
    device: &Device,
    queue: &Queue,
    agent_count: usize,
    speed_min: f32,
    speed_max: f32,
) {
    let agent_buf_size = (agent_count * 4 * std::mem::size_of::<f32>()) as u64;
    let temp_agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Temp Agent Buffer"),
        size: agent_buf_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy from agent_buffer to temp_agent_buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Agent Copy Encoder"),
    });
    encoder.copy_buffer_to_buffer(agent_buffer, 0, &temp_agent_buffer, 0, agent_buf_size);
    queue.submit(Some(encoder.finish()));

    // Map and read
    {
        let agent_slice = temp_agent_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        agent_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();
        let agent_data = agent_slice.get_mapped_range();
        let mut agents: Vec<f32> = bytemuck::cast_slice(&agent_data).to_vec();

        // Reassign speeds
        let speed_range = speed_max - speed_min;
        for i in 0..agent_count {
            let offset = i * 4;
            agents[offset + 3] = speed_min + rand::random::<f32>() * speed_range;
        }
        drop(agent_data);

        // Write back
        queue.write_buffer(agent_buffer, 0, bytemuck::cast_slice(&agents));
    }
}

fn main() {
    let mut settings = Settings::default();
    let mut input = WinitInputHelper::new();

    // Initialize preset manager
    let preset_manager = init_preset_manager();
    let mut current_preset_name = "Default".to_string();

    // Initialize LUT manager and get available LUTs
    let lut_manager = LutManager::new();
    let available_luts = lut_manager.get_available_luts();
    let mut current_lut_index = available_luts
        .iter()
        .position(|name| name == "MATPLOTLIB_bone_r")
        .expect("MATPLOTLIB_bone_r LUT not found");

    // Load initial LUT
    let lut_data = lut_manager
        .load_lut(&available_luts[current_lut_index])
        .expect("Failed to load initial LUT");

    // Initialize the event loop and window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Physarum Simulation")
        .with_inner_size(winit::dpi::LogicalSize::new(
            settings.window_width,
            settings.window_height,
        ))
        .with_fullscreen(if settings.window_fullscreen {
            Some(winit::window::Fullscreen::Borderless(None))
        } else {
            None
        })
        .build(&event_loop)
        .unwrap();

    // Track if LUT is currently reversed
    let mut lut_reversed = false;

    // FPS counter variables
    let mut frame_count = 0;
    let mut last_fps_update = Instant::now();
    let mut last_frame_time = Instant::now();
    let fps_update_interval = Duration::from_secs(1);
    // Help text update interval (30fps)
    let help_update_interval = Duration::from_millis(33); // ~30fps
    let mut last_help_update = Instant::now();
    // LUT name display duration
    let lut_display_duration = Duration::from_secs(3);
    let mut last_lut_update = Instant::now();

    // After creating the window, wrap it in Arc for cheap cloning
    let window = Arc::new(window);

    // Initialize wgpu
    let instance = Instance::new(wgpu::InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });
    let surface = instance.create_surface(&window).unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_buffer_size: u32::MAX as u64,
                max_storage_buffer_binding_size: u32::MAX,
                ..wgpu::Limits::default()
            },
        },
        None,
    ))
    .unwrap();

    // Wrap resources in Arc
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // Get device limits for texture size
    let max_texture_dimension = device.limits().max_texture_dimension_2d;
    info!("Max texture dimension: {}", max_texture_dimension);

    // Use settings for window and simulation parameters
    let logical_width = settings.window_width;
    let logical_height = settings.window_height;
    let agent_count = settings.agent_count;
    let decay_factor = settings.pheromone_decay_factor;

    // Get physical size for HiDPI/Retina displays
    let scale_factor = window.scale_factor();
    let mut physical_width = (logical_width as f64 * scale_factor) as u32;
    let mut physical_height = (logical_height as f64 * scale_factor) as u32;

    // Configure the surface
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats.iter().copied().next().unwrap();
    let mut config = wgpu::SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: physical_width,
        height: physical_height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // Create the simulation state (agent buffer and trail map)
    let mut agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Agent Buffer"),
        size: (agent_count * 4 * std::mem::size_of::<f32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    // Initialize agents with random positions and angles (as f32)
    {
        let mut agent_data = agent_buffer.slice(..).get_mapped_range_mut();
        let agent_f32: &mut [f32] = cast_slice_mut(&mut agent_data);
        for i in 0..agent_count {
            let offset = i * 4;
            agent_f32[offset] = rand::random::<f32>() * physical_width as f32;
            agent_f32[offset + 1] = rand::random::<f32>() * physical_height as f32;
            agent_f32[offset + 2] = rand::random::<f32>() * 2.0 * std::f32::consts::PI;
            let speed_range = settings.agent_speed_max - settings.agent_speed_min;
            agent_f32[offset + 3] = settings.agent_speed_min + rand::random::<f32>() * speed_range;
        }
    }
    agent_buffer.unmap();

    // Create the trail map as a storage buffer instead of a storage texture
    let trail_map_size = (physical_width * physical_height) as usize;
    let mut trail_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Trail Map Buffer"),
        size: (trail_map_size * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create the display texture
    let texture_width = physical_width.min(max_texture_dimension);
    let texture_height = physical_height.min(max_texture_dimension);
    let display_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Display Texture"),
        size: wgpu::Extent3d {
            width: texture_width,
            height: texture_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let display_view = display_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Create a uniform buffer for simulation/display size
    let sim_size_uniform =
        SimSizeUniform::new(physical_width, physical_height, decay_factor, &settings);
    let sim_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sim Size Uniform Buffer"),
        contents: bytemuck::bytes_of(&sim_size_uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Initialize shader and pipeline managers
    let shader_manager = ShaderManager::new(&device);
    let pipeline_manager = PipelineManager::new(&device, &shader_manager);

    // Create LUT buffer
    let mut lut_data_combined = Vec::with_capacity(768);
    lut_data_combined.extend_from_slice(&lut_data.red);
    lut_data_combined.extend_from_slice(&lut_data.green);
    lut_data_combined.extend_from_slice(&lut_data.blue);

    // Convert u8 to u32 for the shader
    let lut_data_u32: Vec<u32> = lut_data_combined.iter().map(|&x| x as u32).collect();

    let lut_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LUT Buffer"),
        contents: bytemuck::cast_slice(&lut_data_u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create a sampler for the display texture
    let display_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Display Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Initialize bind group manager
    let mut bind_group_manager = BindGroupManager::new(
        &device,
        &pipeline_manager.compute_bind_group_layout,
        &pipeline_manager.display_bind_group_layout,
        &pipeline_manager.render_bind_group_layout,
        &agent_buffer,
        &trail_map_buffer,
        &sim_size_buffer,
        &display_view,
        &display_sampler,
        &lut_buffer,
    );

    // Create Arc-wrapped resources for text renderer
    let sim_size_buffer = Arc::new(sim_size_buffer);
    let lut_buffer = Arc::new(lut_buffer);

    // Create text renderer
    let mut text_renderer = TextRenderer::new(
        device.clone(),
        queue.clone(),
        settings.window_height,
        sim_size_buffer.clone(),
        lut_buffer.clone(),
    );

    // Load font
    let font_data = include_bytes!("../Texturina-VariableFont_opsz,wght.ttf");
    let font =
        fontdue::Font::from_bytes(font_data as &[u8], fontdue::FontSettings::default()).unwrap();

    // Main event loop
    let window_for_event = window.clone();
    event_loop
        .run(move |event, target| {
            let window = &window_for_event;
            target.set_control_flow(winit::event_loop::ControlFlow::Poll);

            // Update input helper
            if input.update(&event) {
                // Handle keyboard input
                if input.key_pressed(KeyCode::Escape) {
                    target.exit();
                }

                // Handle modifier keys
                let shift_pressed =
                    input.key_held(KeyCode::ShiftLeft) || input.key_held(KeyCode::ShiftRight);

                // Handle key combinations for settings adjustment
                if input.key_held(KeyCode::KeyT) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 0.01 } else { 0.1 };
                        settings.agent_turn_speed =
                            (settings.agent_turn_speed + increment).min(6.283_185_5);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 0.01 } else { 0.1 };
                        settings.agent_turn_speed =
                            (settings.agent_turn_speed - decrement).max(0.0);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyJ) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 0.01 } else { 0.1 };
                        settings.agent_jitter += increment;
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 0.01 } else { 0.1 };
                        settings.agent_jitter = (settings.agent_jitter - decrement).max(0.0);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyS) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 1.0 } else { 5.0 };
                        settings.agent_speed_min += increment;
                        reassign_agent_speeds(
                            &agent_buffer,
                            &device,
                            &queue,
                            settings.agent_count,
                            settings.agent_speed_min,
                            settings.agent_speed_max,
                        );
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 1.0 } else { 5.0 };
                        settings.agent_speed_min = (settings.agent_speed_min - decrement).max(0.0);
                        reassign_agent_speeds(
                            &agent_buffer,
                            &device,
                            &queue,
                            settings.agent_count,
                            settings.agent_speed_min,
                            settings.agent_speed_max,
                        );
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyX) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 1.0 } else { 5.0 };
                        settings.agent_speed_max += increment;
                        reassign_agent_speeds(
                            &agent_buffer,
                            &device,
                            &queue,
                            settings.agent_count,
                            settings.agent_speed_min,
                            settings.agent_speed_max,
                        );
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 1.0 } else { 5.0 };
                        settings.agent_speed_max =
                            (settings.agent_speed_max - decrement).max(settings.agent_speed_min);
                        reassign_agent_speeds(
                            &agent_buffer,
                            &device,
                            &queue,
                            settings.agent_count,
                            settings.agent_speed_min,
                            settings.agent_speed_max,
                        );
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyA) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 0.01 } else { 0.1 };
                        settings.agent_sensor_angle += increment;
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 0.01 } else { 0.1 };
                        settings.agent_sensor_angle =
                            (settings.agent_sensor_angle - decrement).max(0.0);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyD) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 1.0 } else { 5.0 };
                        settings.agent_sensor_distance += increment;
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 1.0 } else { 5.0 };
                        settings.agent_sensor_distance =
                            (settings.agent_sensor_distance - decrement).max(0.0);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyR) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 0.1 } else { 1.0 };
                        settings.pheromone_deposition_amount += increment;
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 0.1 } else { 1.0 };
                        settings.pheromone_deposition_amount =
                            (settings.pheromone_deposition_amount - decrement).max(0.0);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyV) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 0.1 } else { 1.0 };
                        settings.pheromone_decay_factor += increment;
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 0.1 } else { 1.0 };
                        settings.pheromone_decay_factor =
                            (settings.pheromone_decay_factor - decrement).max(0.0);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyB) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 0.01 } else { 0.1 };
                        settings.pheromone_diffusion_rate =
                            (settings.pheromone_diffusion_rate + increment).min(1.0);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 0.01 } else { 0.1 };
                        settings.pheromone_diffusion_rate =
                            (settings.pheromone_diffusion_rate - decrement).max(0.0);
                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                if input.key_held(KeyCode::KeyN) {
                    if input.key_pressed(KeyCode::ArrowUp) {
                        let increment = if shift_pressed { 100_000 } else { 1_000_000 };
                        settings.agent_count += increment;
                        // Resize agent buffer
                        let agent_buf_size =
                            (settings.agent_count * 4 * std::mem::size_of::<f32>()) as u64;
                        let new_agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Agent Buffer"),
                            size: agent_buf_size,
                            usage: BufferUsages::STORAGE
                                | BufferUsages::COPY_SRC
                                | BufferUsages::COPY_DST,
                            mapped_at_creation: true,
                        });
                        // Initialize new agents with random positions and angles
                        {
                            let mut agent_data = new_agent_buffer.slice(..).get_mapped_range_mut();
                            let agent_f32: &mut [f32] = cast_slice_mut(&mut agent_data);
                            for i in 0..settings.agent_count {
                                let offset = i * 4;
                                agent_f32[offset] = rand::random::<f32>() * physical_width as f32;
                                agent_f32[offset + 1] =
                                    rand::random::<f32>() * physical_height as f32;
                                agent_f32[offset + 2] =
                                    rand::random::<f32>() * 2.0 * std::f32::consts::PI;
                                let speed_range =
                                    settings.agent_speed_max - settings.agent_speed_min;
                                agent_f32[offset + 3] =
                                    settings.agent_speed_min + rand::random::<f32>() * speed_range;
                            }
                        }
                        new_agent_buffer.unmap();

                        // Update bind groups with new agent buffer
                        bind_group_manager.update_compute_bind_group(
                            &device,
                            &pipeline_manager.compute_bind_group_layout,
                            &new_agent_buffer,
                            &trail_map_buffer,
                            &sim_size_buffer,
                        );

                        // Replace old buffer with new one
                        agent_buffer = new_agent_buffer;

                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    } else if input.key_pressed(KeyCode::ArrowDown) {
                        let decrement = if shift_pressed { 100_000 } else { 1_000_000 };
                        settings.agent_count =
                            (settings.agent_count.saturating_sub(decrement)).max(1);
                        // Resize agent buffer
                        let agent_buf_size =
                            (settings.agent_count * 4 * std::mem::size_of::<f32>()) as u64;
                        let new_agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Agent Buffer"),
                            size: agent_buf_size,
                            usage: BufferUsages::STORAGE
                                | BufferUsages::COPY_SRC
                                | BufferUsages::COPY_DST,
                            mapped_at_creation: true,
                        });
                        // Initialize new agents with random positions and angles
                        {
                            let mut agent_data = new_agent_buffer.slice(..).get_mapped_range_mut();
                            let agent_f32: &mut [f32] = cast_slice_mut(&mut agent_data);
                            for i in 0..settings.agent_count {
                                let offset = i * 4;
                                agent_f32[offset] = rand::random::<f32>() * physical_width as f32;
                                agent_f32[offset + 1] =
                                    rand::random::<f32>() * physical_height as f32;
                                agent_f32[offset + 2] =
                                    rand::random::<f32>() * 2.0 * std::f32::consts::PI;
                                let speed_range =
                                    settings.agent_speed_max - settings.agent_speed_min;
                                agent_f32[offset + 3] =
                                    settings.agent_speed_min + rand::random::<f32>() * speed_range;
                            }
                        }
                        new_agent_buffer.unmap();

                        // Update bind groups with new agent buffer
                        bind_group_manager.update_compute_bind_group(
                            &device,
                            &pipeline_manager.compute_bind_group_layout,
                            &new_agent_buffer,
                            &trail_map_buffer,
                            &sim_size_buffer,
                        );

                        // Replace old buffer with new one
                        agent_buffer = new_agent_buffer;

                        update_settings(
                            &mut settings,
                            &mut current_preset_name,
                            &sim_size_buffer,
                            &queue,
                            physical_width,
                            physical_height,
                        );
                    }
                }

                // Handle preset cycling
                if input.key_pressed(KeyCode::KeyP) {
                    // Get all preset names from the preset manager
                    let preset_names = preset_manager.get_preset_names();

                    // Find current index
                    let current_index = preset_names
                        .iter()
                        .position(|name| name == &current_preset_name)
                        .unwrap_or(0);

                    // Calculate new index
                    let new_index = if shift_pressed {
                        if current_index == 0 {
                            preset_names.len() - 1
                        } else {
                            current_index - 1
                        }
                    } else {
                        (current_index + 1) % preset_names.len()
                    };

                    // Apply new preset
                    if let Some(preset) = preset_manager.get_preset(&preset_names[new_index]) {
                        settings = preset.settings.clone();
                        current_preset_name = preset.name.clone();

                        // Update uniform buffer with new settings
                        let sim_size_uniform = SimSizeUniform::new(
                            physical_width,
                            physical_height,
                            settings.pheromone_decay_factor,
                            &settings,
                        );
                        queue.write_buffer(
                            &sim_size_buffer,
                            0,
                            bytemuck::bytes_of(&sim_size_uniform),
                        );
                    }
                }

                // Handle LUT cycling
                if input.key_pressed(KeyCode::KeyG) {
                    if shift_pressed {
                        if current_lut_index == 0 {
                            current_lut_index = available_luts.len() - 1;
                        } else {
                            current_lut_index -= 1;
                        }
                    } else {
                        current_lut_index = (current_lut_index + 1) % available_luts.len();
                    }

                    let mut lut_data = lut_manager
                        .load_lut(&available_luts[current_lut_index])
                        .expect("Failed to load LUT");

                    if lut_reversed {
                        lut_data.reverse();
                    }

                    let mut lut_data_combined = Vec::with_capacity(768);
                    lut_data_combined.extend_from_slice(&lut_data.red);
                    lut_data_combined.extend_from_slice(&lut_data.green);
                    lut_data_combined.extend_from_slice(&lut_data.blue);
                    let lut_data_u32: Vec<u32> =
                        lut_data_combined.iter().map(|&x| x as u32).collect();
                    queue.write_buffer(&lut_buffer, 0, bytemuck::cast_slice(&lut_data_u32));

                    window.set_title(&format!(
                        "Physarum Simulation - LUT: {}{}",
                        available_luts[current_lut_index],
                        if lut_reversed { " (Reversed)" } else { "" }
                    ));
                    last_lut_update = Instant::now();
                }

                // Handle LUT reversal (changed from R to F)
                if input.key_pressed(KeyCode::KeyF) {
                    let mut lut_data = lut_manager
                        .load_lut(&available_luts[current_lut_index])
                        .expect("Failed to load LUT");

                    if lut_reversed {
                        lut_data = lut_manager
                            .load_lut(&available_luts[current_lut_index])
                            .expect("Failed to load LUT");
                    } else {
                        lut_data.reverse();
                    }

                    lut_reversed = !lut_reversed;

                    let mut lut_data_combined = Vec::with_capacity(768);
                    lut_data_combined.extend_from_slice(&lut_data.red);
                    lut_data_combined.extend_from_slice(&lut_data.green);
                    lut_data_combined.extend_from_slice(&lut_data.blue);
                    let lut_data_u32: Vec<u32> =
                        lut_data_combined.iter().map(|&x| x as u32).collect();
                    queue.write_buffer(&lut_buffer, 0, bytemuck::cast_slice(&lut_data_u32));

                    window.set_title(&format!(
                        "Physarum Simulation - LUT: {}{}",
                        available_luts[current_lut_index],
                        if lut_reversed { " (Reversed)" } else { "" }
                    ));
                    last_lut_update = Instant::now();
                }

                // Handle help text toggle
                if input.key_pressed(KeyCode::Slash) {
                    text_renderer.toggle_visibility();
                }

                // Handle trail map clear
                if input.key_pressed(KeyCode::KeyC) {
                    let trail_map_size = (physical_width * physical_height) as usize;
                    let clear_buffer = vec![0.0f32; trail_map_size];
                    queue.write_buffer(&trail_map_buffer, 0, bytemuck::cast_slice(&clear_buffer));
                }
            }

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => target.exit(),
                Event::WindowEvent {
                    event: WindowEvent::Resized(new_size),
                    ..
                } => {
                    // Store old sizes
                    let old_width = config.width;
                    let old_height = config.height;

                    // Get the current scale factor
                    let scale_factor = window.scale_factor();

                    // Calculate logical size from physical size
                    let logical_width = (new_size.width as f64 / scale_factor) as u32;
                    let logical_height = (new_size.height as f64 / scale_factor) as u32;

                    // Calculate physical size for HiDPI/Retina displays
                    physical_width = (logical_width as f64 * scale_factor) as u32;
                    physical_height = (logical_height as f64 * scale_factor) as u32;

                    info!(
                        "[resize] logical size: {}x{}, physical size: {}x{}, scale factor: {}",
                        logical_width,
                        logical_height,
                        physical_width,
                        physical_height,
                        scale_factor
                    );

                    // Update surface configuration
                    config.width = physical_width;
                    config.height = physical_height;
                    surface.configure(&device, &config);

                    // --- SCALE AGENT POSITIONS ---
                    {
                        let agent_buf_size = (agent_count * 4 * std::mem::size_of::<f32>()) as u64;
                        let temp_agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Temp Agent Buffer"),
                            size: agent_buf_size,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        // Copy from agent_buffer to temp_agent_buffer
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Agent Copy Encoder"),
                            });
                        encoder.copy_buffer_to_buffer(
                            &agent_buffer,
                            0,
                            &temp_agent_buffer,
                            0,
                            agent_buf_size,
                        );
                        queue.submit(Some(encoder.finish()));
                        // Map and read
                        {
                            let agent_slice = temp_agent_buffer.slice(..);
                            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
                            agent_slice
                                .map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
                            device.poll(wgpu::Maintain::Wait);
                            receiver.recv().unwrap().unwrap();
                            let agent_data = agent_slice.get_mapped_range();
                            let mut agents: Vec<f32> = bytemuck::cast_slice(&agent_data).to_vec();
                            // Scale positions
                            for i in 0..agent_count {
                                let offset = i * 4;
                                agents[offset] *= physical_width as f32 / old_width as f32;
                                agents[offset + 1] *= physical_height as f32 / old_height as f32;
                            }
                            drop(agent_data);
                            // Write back
                            queue.write_buffer(&agent_buffer, 0, bytemuck::cast_slice(&agents));
                        }
                        // temp_agent_buffer drops here
                    }

                    // --- ALWAYS RESIZE PHEROMONE/TRAIL MAP TO MATCH WINDOW ---
                    {
                        let new_size = (physical_width * physical_height) as usize;
                        let trail_buf_size = (new_size * std::mem::size_of::<f32>()) as u64;
                        // Create new trail map buffer with the correct size
                        let new_trail_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Trail Map Buffer"),
                            size: trail_buf_size,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        });
                        // Optionally, resample old data if desired (omitted for clarity)
                        // Replace old buffer with new one
                        trail_map_buffer = new_trail_map_buffer;
                    }

                    // Recreate the display texture
                    let texture_width = physical_width.min(max_texture_dimension);
                    let texture_height = physical_height.min(max_texture_dimension);
                    info!(
                        "[resize] texture_width: {}, texture_height: {}, max_texture_dimension: {}",
                        texture_width, texture_height, max_texture_dimension
                    );
                    info!(
                        "[resize] display_texture size: width: {}, height: {}",
                        texture_width, texture_height
                    );

                    // Create new display texture with updated size
                    let display_texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Display Texture"),
                        size: wgpu::Extent3d {
                            width: texture_width,
                            height: texture_height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::STORAGE_BINDING
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    });
                    let display_view =
                        display_texture.create_view(&wgpu::TextureViewDescriptor::default());

                    // Update uniform buffer with new size
                    let sim_size_uniform = SimSizeUniform::new(
                        physical_width,
                        physical_height,
                        decay_factor,
                        &settings,
                    );
                    queue.write_buffer(&sim_size_buffer, 0, bytemuck::bytes_of(&sim_size_uniform));

                    // Update bind groups with new resources
                    bind_group_manager.update_compute_bind_group(
                        &device,
                        &pipeline_manager.compute_bind_group_layout,
                        &agent_buffer,
                        &trail_map_buffer,
                        &sim_size_buffer,
                    );

                    bind_group_manager.update_display_bind_group(
                        &device,
                        &pipeline_manager.display_bind_group_layout,
                        &trail_map_buffer,
                        &display_view,
                        &sim_size_buffer,
                        &lut_buffer,
                    );

                    bind_group_manager.update_render_bind_group(
                        &device,
                        &pipeline_manager.render_bind_group_layout,
                        &display_view,
                        &display_sampler,
                    );

                    // Update text renderer window size
                    text_renderer.update_window_size(physical_height);
                }
                Event::AboutToWait => {
                    // Calculate FPS
                    frame_count += 1;
                    let current_time = Instant::now();
                    let elapsed = current_time - last_fps_update;

                    if elapsed >= fps_update_interval {
                        let fps = frame_count as f64 / elapsed.as_secs_f64();
                        let frame_time = (current_time - last_frame_time).as_secs_f64() * 1000.0;

                        // Only update title with FPS if LUT name display duration has passed
                        if current_time - last_lut_update >= lut_display_duration {
                            window.set_title(&format!(
                                "Physarum Simulation - FPS: {:.1} ({:.1}ms)",
                                fps, frame_time
                            ));
                        }
                        frame_count = 0;
                        last_fps_update = current_time;
                    }
                    last_frame_time = current_time;

                    // Update help text at 30fps
                    let current_time = Instant::now();
                    if current_time - last_help_update >= help_update_interval {
                        // Calculate current FPS and frame time
                        let fps =
                            frame_count as f64 / (current_time - last_fps_update).as_secs_f64();
                        let frame_time = (current_time - last_frame_time).as_secs_f64() * 1000.0;

                        // Update help text
                        let help_text = {
                            let decay_factor =
                                format_float_dynamic(settings.pheromone_decay_factor);
                            let diffusion_rate =
                                format_float_dynamic(settings.pheromone_diffusion_rate);
                            let deposition_amount =
                                format_float_dynamic(settings.pheromone_deposition_amount);
                            let agent_count = settings.agent_count.to_formatted_string(&Locale::en);
                            let agent_jitter = format_float_dynamic(settings.agent_jitter);
                            let agent_speed_min = format_float_dynamic(settings.agent_speed_min);
                            let agent_speed_max = format_float_dynamic(settings.agent_speed_max);
                            let agent_turn_speed = format_float_dynamic(
                                settings.agent_turn_speed * 180.0 / std::f32::consts::PI,
                            );
                            let agent_sensor_angle = format_float_dynamic(
                                settings.agent_sensor_angle * 180.0 / std::f32::consts::PI,
                            );
                            let agent_sensor_distance =
                                format_float_dynamic(settings.agent_sensor_distance);

                            format!(
                                "FPS:\t{fps:.1} ({frame_time:.1}ms)\n\
                                (P) Preset:\t{current_preset_name}\n\
                                (N) Agents:\t{agent_count}\n\
                                (V) Decay:\t{decay_factor}\n\
                                (B) Diffusion:\t{diffusion_rate}\n\
                                (R) Deposition:\t{deposition_amount}\n\
                                (J) Jitter:\t{agent_jitter}\n\
                                (S) Min Speed:\t{agent_speed_min}\n\
                                (X) Max Speed:\t{agent_speed_max}\n\
                                (T) Turn:\t{agent_turn_speed}°\n\
                                (A) Angle:\t{agent_sensor_angle}°\n\
                                (D) Distance:\t{agent_sensor_distance}\n\
                                Press / to toggle help\n\
                                Press C to clear trail map\n\
                                Press G to cycle LUTs (Shift+G for reverse)\n\
                                Press F to toggle LUT reversal\n\
                                Hold any key + arrows to adjust its setting\n\
                                Adjust with Shift held for fine control",
                            )
                        };

                        text_renderer.render_text(&help_text, &font, window.inner_size());
                        last_help_update = current_time;
                    }

                    // Run the compute pass to update agents and trail map
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: None,
                                timestamp_writes: None,
                            });
                        compute_pass.set_pipeline(&pipeline_manager.compute_pipeline);
                        compute_pass.set_bind_group(0, &bind_group_manager.compute_bind_group, &[]);
                        // Split the workgroups across multiple dimensions
                        let workgroup_size = 64u32;
                        let total_workgroups = (agent_count as u32).div_ceil(workgroup_size);
                        let workgroups_x = total_workgroups.min(65535);
                        let workgroups_y = total_workgroups.div_ceil(workgroups_x);
                        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
                    }
                    queue.submit(Some(encoder.finish()));

                    // Run the decay pass
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut decay_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Decay Pass"),
                                timestamp_writes: None,
                            });
                        decay_pass.set_pipeline(&pipeline_manager.decay_pipeline);
                        decay_pass.set_bind_group(0, &bind_group_manager.compute_bind_group, &[]);
                        let workgroup_size = 16u32;
                        let dispatch_x = physical_width.div_ceil(workgroup_size);
                        let dispatch_y = physical_height.div_ceil(workgroup_size);
                        decay_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                    }
                    queue.submit(Some(encoder.finish()));

                    // Run the display compute pass
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: None,
                                timestamp_writes: None,
                            });
                        compute_pass.set_pipeline(&pipeline_manager.display_pipeline);
                        compute_pass.set_bind_group(0, &bind_group_manager.display_bind_group, &[]);
                        let workgroup_size = 16u32;
                        let dispatch_x = physical_width.div_ceil(workgroup_size);
                        let dispatch_y = physical_height.div_ceil(workgroup_size);
                        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                    }
                    queue.submit(Some(encoder.finish()));

                    // Run diffusion
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    let mut diffuse_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Diffuse Pass"),
                            timestamp_writes: None,
                        });
                    diffuse_pass.set_pipeline(&pipeline_manager.diffuse_pipeline);
                    diffuse_pass.set_bind_group(0, &bind_group_manager.compute_bind_group, &[]);
                    diffuse_pass.dispatch_workgroups(
                        (physical_width + 15) / 16,
                        (physical_height + 15) / 16,
                        1,
                    );
                    drop(diffuse_pass);
                    queue.submit(Some(encoder.finish()));

                    // Render the trail map to the screen
                    let frame = surface.get_current_texture().unwrap();
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                        render_pass.set_pipeline(&pipeline_manager.render_pipeline);
                        render_pass.set_bind_group(0, &bind_group_manager.render_bind_group, &[]);
                        render_pass.draw(0..6, 0..1);

                        // Draw text overlay
                        if let Some(text_bind_group) = text_renderer.get_bind_group() {
                            render_pass.set_pipeline(&pipeline_manager.text_pipeline);
                            render_pass.set_bind_group(0, text_bind_group, &[]);
                            render_pass.draw(0..6, 0..1);
                        }
                    }
                    queue.submit(Some(encoder.finish()));

                    frame.present();
                }
                _ => {}
            }
        })
        .expect("Error in event loop");
}
