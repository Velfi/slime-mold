use bytemuck::cast_slice_mut;
use slime_mold::egui_tools::EguiRenderer;
use slime_mold::lut_manager::LutManager;
use slime_mold::render::{
    bind_group_manager::BindGroupManager, pipeline_manager::PipelineManager,
    shader_manager::ShaderManager, text_renderer::TextRenderer,
};
use slime_mold::settings::Settings;
use slime_mold::simulation::SimSizeUniform;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::debug;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowId};

use slime_mold::presets::init_preset_manager;
use wgpu::util::DeviceExt;
use wgpu::{Backends, Buffer, BufferUsages, Device, Instance, Queue, TextureUsages};

// Helper function to update settings and sync to GPU
fn update_settings(
    settings: &Settings,
    sim_size_buffer: &Buffer,
    queue: &Queue,
    physical_width: u32,
    physical_height: u32,
) {
    let sim_size_uniform = SimSizeUniform::new(
        physical_width,
        physical_height,
        settings.pheromone_decay_rate,
        settings,
    );
    queue.write_buffer(sim_size_buffer, 0, bytemuck::bytes_of(&sim_size_uniform));
}

// Helper function to reassign agent speeds when speed settings change using GPU compute
fn reassign_agent_speeds_gpu(
    device: &Device,
    queue: &Queue,
    pipeline_manager: &PipelineManager,
    bind_group_manager: &BindGroupManager,
    agent_count: usize,
) {
    // Create a simple compute shader dispatch to update speeds on GPU
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Agent Speed Update Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Agent Speed Update Pass"),
            timestamp_writes: None,
        });

        // Use the dedicated speed update pipeline
        cpass.set_pipeline(&pipeline_manager.speed_update_pipeline);
        cpass.set_bind_group(0, &bind_group_manager.compute_bind_group, &[]);
        cpass.dispatch_workgroups((agent_count as u32).div_ceil(256).min(65535), 1, 1);
    }

    queue.submit(Some(encoder.finish()));
}

// Helper function to create new agent buffer with given count
fn create_agent_buffer(
    device: &Device,
    agent_count: usize,
    physical_width: u32,
    physical_height: u32,
    settings: &Settings,
) -> Buffer {
    let agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Agent Buffer"),
        size: (agent_count * 4 * std::mem::size_of::<f32>()) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    // Initialize agents with random positions and angles
    {
        let mut agent_data = agent_buffer.slice(..).get_mapped_range_mut();
        let agent_f32: &mut [f32] = cast_slice_mut(&mut agent_data);
        for i in 0..agent_count {
            let offset = i * 4;
            agent_f32[offset] = rand::random::<f32>() * physical_width as f32;
            agent_f32[offset + 1] = rand::random::<f32>() * physical_height as f32;
            // Use the agent_possible_starting_headings range from settings
            let heading_range = settings.agent_possible_starting_headings.end
                - settings.agent_possible_starting_headings.start;
            let heading_radians = (settings.agent_possible_starting_headings.start
                + rand::random::<f32>() * heading_range)
                * std::f32::consts::PI
                / 180.0;
            agent_f32[offset + 2] = heading_radians;
            let speed_range = settings.agent_speed_max - settings.agent_speed_min;
            agent_f32[offset + 3] = settings.agent_speed_min + rand::random::<f32>() * speed_range;
        }
    }
    agent_buffer.unmap();
    agent_buffer
}

// Helper function to reset trail map to zero
fn reset_trails(
    trail_map_buffer: &Buffer,
    queue: &Queue,
    physical_width: u32,
    physical_height: u32,
) {
    // Use the same dimensions as the buffer creation
    let trail_map_size = (physical_width * physical_height) as usize;
    let clear_buffer = vec![0.0f32; trail_map_size];
    queue.write_buffer(trail_map_buffer, 0, bytemuck::cast_slice(&clear_buffer));
}

// Helper function to reset agents (remove all and spawn new ones)
fn reset_agents(
    agent_buffer: &Buffer,
    queue: &Queue,
    physical_width: u32,
    physical_height: u32,
    settings: &Settings,
    agent_count: usize,
) {
    // Generate completely new agent data directly into a vector
    let mut agent_data = Vec::with_capacity(agent_count * 4);
    for _i in 0..agent_count {
        // New random position
        agent_data.push(rand::random::<f32>() * physical_width as f32);
        agent_data.push(rand::random::<f32>() * physical_height as f32);

        // New random heading
        let heading_range = settings.agent_possible_starting_headings.end
            - settings.agent_possible_starting_headings.start;
        let heading_radians = (settings.agent_possible_starting_headings.start
            + rand::random::<f32>() * heading_range)
            * std::f32::consts::PI
            / 180.0;
        agent_data.push(heading_radians);

        // New random speed
        let speed_range = settings.agent_speed_max - settings.agent_speed_min;
        agent_data.push(settings.agent_speed_min + rand::random::<f32>() * speed_range);
    }

    // Use write_buffer to completely replace all agent data
    // This is non-blocking and more efficient
    queue.write_buffer(agent_buffer, 0, bytemuck::cast_slice(&agent_data));
}

// Helper function to randomize settings while preserving agent count
fn randomize_settings(settings: &mut Settings, agent_count: usize) -> usize {
    // Store current agent count
    let current_agent_count = agent_count;

    // Randomize all settings
    // Use high range for decay rate to allow for more variation
    settings.pheromone_decay_rate = rand::random::<f32>() * 10.0; // 1.0 is normal value
    settings.pheromone_deposition_rate = rand::random::<f32>() * 100.0 / 100.0; // Convert to percentage
    settings.pheromone_diffusion_rate = rand::random::<f32>() * 100.0 / 100.0; // Convert to percentage
    settings.agent_speed_min = rand::random::<f32>() * 500.0;
    settings.agent_speed_max =
        settings.agent_speed_min + rand::random::<f32>() * (500.0 - settings.agent_speed_min);
    settings.agent_turn_rate = (rand::random::<f32>() * 360.0) * std::f32::consts::PI / 180.0; // Convert degrees to radians
    settings.agent_jitter = rand::random::<f32>() * 5.0;
    settings.agent_sensor_angle = (rand::random::<f32>() * 180.0) * std::f32::consts::PI / 180.0; // Convert degrees to radians
    settings.agent_sensor_distance = rand::random::<f32>() * 500.0;

    // Randomize gradient settings
    settings.gradient_enabled = rand::random::<bool>();
    let gradient_types = slime_mold::settings::GradientType::all();
    settings.gradient_type =
        gradient_types[(rand::random::<u32>() as usize) % gradient_types.len()];
    settings.gradient_strength = rand::random::<f32>() * 100.0;
    settings.gradient_center_x = rand::random::<f32>();
    settings.gradient_center_y = rand::random::<f32>();
    settings.gradient_size = 0.1 + rand::random::<f32>() * 1.9;
    settings.gradient_angle = rand::random::<f32>() * 360.0;

    // Randomize starting direction range
    let start = rand::random::<f32>() * 360.0;
    let end = start + rand::random::<f32>() * (360.0 - start);
    settings.agent_possible_starting_headings = start..end;

    current_agent_count
}

struct App {
    // Window and graphics state
    window: Option<Arc<Window>>,
    instance: Option<Instance>,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
    config: Option<wgpu::SurfaceConfiguration>,

    // Simulation settings and state
    settings: Settings,
    settings_changed: bool,
    needs_gpu_update: bool,
    needs_display_update: bool,
    ui_visible: bool,
    paused: bool,
    decay_rate_hi_range: bool,
    settings_have_changed: bool,

    // FPS tracking
    frame_times: Vec<Duration>,
    last_frame_time: Instant,

    // Rendering components
    egui_renderer: Option<EguiRenderer>,
    bind_group_manager: Option<BindGroupManager>,
    pipeline_manager: Option<PipelineManager>,
    text_renderer: Option<TextRenderer>,

    // Buffers and textures
    agent_buffer: Option<Buffer>,
    trail_map_buffer: Option<Buffer>,
    gradient_buffer: Option<Buffer>,
    sim_size_buffer: Option<Arc<Buffer>>,
    lut_buffer: Option<Arc<Buffer>>,
    display_texture: Option<wgpu::Texture>,
    display_view: Option<wgpu::TextureView>,
    display_sampler: Option<wgpu::Sampler>,

    // LUT management
    current_lut_index: usize,
    previous_lut_index: usize,
    lut_reversed: bool,
    lut_preview_cache: HashMap<(String, bool), Vec<egui::Color32>>,
    available_luts: Vec<String>,
    lut_manager: LutManager,

    // Presets
    preset_manager: slime_mold::presets::PresetManager,
    preset_names: Vec<String>,
    selected_preset: String,
    new_preset_name: String,
    save_preset_dialog_open: bool,
    agent_count: usize,
    previous_agent_count: usize,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            // Create window
            let mut attributes = Window::default_attributes()
                .with_title("Physarum Simulation")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    self.settings.window_width,
                    self.settings.window_height,
                ));
            if self.settings.window_fullscreen {
                attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
            }

            let window = Arc::new(event_loop.create_window(attributes).unwrap());

            // Initialize wgpu
            let instance = Instance::new(&wgpu::InstanceDescriptor {
                backends: Backends::all(),
                ..Default::default()
            });

            // Create surface using Arc<Window> - this eliminates lifetime issues
            let surface = instance.create_surface(window.clone()).unwrap();

            // Store the window after creating surface
            self.window = Some(window);

            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                }))
                .unwrap();

            let adapter_limits = adapter.limits();
            debug!(
                "Adapter max buffer size: {}",
                adapter_limits.max_buffer_size
            );
            debug!(
                "Adapter max storage buffer binding size: {}",
                adapter_limits.max_storage_buffer_binding_size
            );

            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    memory_hints: wgpu::MemoryHints::default(),
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_buffer_size: adapter_limits.max_buffer_size,
                        max_storage_buffer_binding_size:
                            adapter_limits.max_storage_buffer_binding_size,
                        ..wgpu::Limits::default()
                    },
                },
                None,
            ))
            .unwrap();

            let device = Arc::new(device);
            let queue = Arc::new(queue);

            // Get device limits for texture size
            let max_texture_dimension = device.limits().max_texture_dimension_2d;
            debug!("Max texture dimension: {}", max_texture_dimension);

            // Calculate max agents based on device buffer limits
            let max_agents = (device.limits().max_buffer_size
                / (4 * std::mem::size_of::<f32>() as u64)) as usize;
            debug!("Max agents based on device limits: {}", max_agents);

            // Use settings for window and simulation parameters
            let logical_width = self.settings.window_width;
            let logical_height = self.settings.window_height;

            // Get physical size for HiDPI/Retina displays
            let scale_factor = self.window.as_ref().unwrap().scale_factor();
            let physical_width = (logical_width as f64 * scale_factor) as u32;
            let physical_height = (logical_height as f64 * scale_factor) as u32;

            // Configure the surface
            let surface_caps = surface.get_capabilities(&adapter);
            let surface_format = surface_caps
                .formats
                .iter()
                .copied()
                .find(|f| !f.is_srgb())
                .unwrap_or(surface_caps.formats[0]);
            let config = wgpu::SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: physical_width,
                height: physical_height,
                present_mode: surface_caps.present_modes[0],
                alpha_mode: wgpu::CompositeAlphaMode::PostMultiplied,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            };
            surface.configure(&device, &config);

            // Create the simulation state (agent buffer and trail map)
            let agent_buffer = create_agent_buffer(
                &device,
                self.agent_count,
                physical_width,
                physical_height,
                &self.settings,
            );

            // Create the trail map as a storage buffer instead of a storage texture
            let trail_map_size = (physical_width * physical_height) as usize;
            let trail_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Trail Map Buffer"),
                size: (trail_map_size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Create the gradient buffer for constant pheromone gradients
            let gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gradient Buffer"),
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
            let sim_size_uniform = SimSizeUniform::new(
                physical_width,
                physical_height,
                self.settings.pheromone_decay_rate,
                &self.settings,
            );
            let sim_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sim Size Uniform Buffer"),
                contents: bytemuck::bytes_of(&sim_size_uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // Initialize shader and pipeline managers
            let shader_manager = ShaderManager::new(&device);
            let pipeline_manager = PipelineManager::new(&device, &shader_manager);

            // Load LUT
            let lut_data = self
                .lut_manager
                .load_lut(&self.available_luts[self.current_lut_index])
                .expect("Failed to load initial LUT");

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
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

            // Initialize bind group manager
            let bind_group_manager = BindGroupManager::new(
                &device,
                &pipeline_manager.compute_bind_group_layout,
                &pipeline_manager.gradient_bind_group_layout,
                &pipeline_manager.display_bind_group_layout,
                &pipeline_manager.render_bind_group_layout,
                &agent_buffer,
                &trail_map_buffer,
                &gradient_buffer,
                &sim_size_buffer,
                &display_view,
                &display_sampler,
                &lut_buffer,
            );

            // Create Arc-wrapped resources for text renderer
            let sim_size_buffer = Arc::new(sim_size_buffer);
            let lut_buffer = Arc::new(lut_buffer);

            // Create text renderer
            let text_renderer = TextRenderer::new(
                device.clone(),
                queue.clone(),
                self.settings.window_height,
                sim_size_buffer.clone(),
                lut_buffer.clone(),
            );

            // Create egui renderer
            let egui_renderer = EguiRenderer::new(
                device.as_ref(),
                surface_format,
                None,
                1,
                self.window.as_ref().unwrap(),
            );

            // Set dark theme for egui
            egui_renderer.context().set_visuals(egui::Visuals::dark());

            // Store all the initialized state (window already stored above)
            self.instance = Some(instance);
            self.surface = Some(surface);
            self.adapter = Some(adapter);
            self.device = Some(device);
            self.queue = Some(queue);
            self.config = Some(config);
            self.agent_buffer = Some(agent_buffer);
            self.trail_map_buffer = Some(trail_map_buffer);
            self.gradient_buffer = Some(gradient_buffer);
            self.sim_size_buffer = Some(sim_size_buffer);
            self.lut_buffer = Some(lut_buffer);
            self.display_texture = Some(display_texture);
            self.display_view = Some(display_view);
            self.display_sampler = Some(display_sampler);
            self.bind_group_manager = Some(bind_group_manager);
            self.pipeline_manager = Some(pipeline_manager);
            self.text_renderer = Some(text_renderer);
            self.egui_renderer = Some(egui_renderer);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(window) = &self.window {
            // Handle egui input
            if let Some(egui_renderer) = &mut self.egui_renderer {
                egui_renderer.handle_input(window, &event);
            }

            // Handle key press for UI toggle
            if let WindowEvent::KeyboardInput {
                event: key_event, ..
            } = &event
            {
                if key_event.state.is_pressed() {
                    if let winit::keyboard::Key::Character(c) = &key_event.logical_key {
                        if c == "/" {
                            self.ui_visible = !self.ui_visible;
                        } else if c == "r" {
                            self.agent_count =
                                randomize_settings(&mut self.settings, self.agent_count);

                            // Mark settings as changed
                            self.settings_changed = true;
                            self.needs_display_update = true;

                            // Check if settings still match current preset
                            if self.preset_manager.get_preset(&self.selected_preset).is_some() {
                                self.settings_have_changed = true;  // Settings changed, mark as unsaved
                            }
                        }
                    }
                }
            }
        }

        match event {
            WindowEvent::Resized(physical_size) => {
                if let (Some(surface), Some(device), Some(config)) =
                    (&self.surface, &self.device, &mut self.config)
                {
                    // Update surface configuration
                    config.width = physical_size.width;
                    config.height = physical_size.height;
                    surface.configure(device, config);

                    // Update simulation size and settings
                    self.settings.window_width = physical_size.width;
                    self.settings.window_height = physical_size.height;
                    update_settings(
                        &self.settings,
                        self.sim_size_buffer.as_ref().unwrap(),
                        self.queue.as_ref().unwrap(),
                        physical_size.width,
                        physical_size.height,
                    );

                    // Recreate trail map buffer with new dimensions
                    let trail_map_size = (physical_size.width * physical_size.height) as usize;
                    self.trail_map_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Trail Map Buffer"),
                        size: (trail_map_size * std::mem::size_of::<f32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));

                    // Recreate agent buffer with new dimensions
                    self.agent_buffer = Some(create_agent_buffer(
                        device,
                        self.agent_count,
                        physical_size.width,
                        physical_size.height,
                        &self.settings,
                    ));

                    // Recreate display texture with new dimensions
                    let max_texture_dimension = device.limits().max_texture_dimension_2d;
                    let texture_width = physical_size.width.min(max_texture_dimension);
                    let texture_height = physical_size.height.min(max_texture_dimension);
                    self.display_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
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
                    }));
                    self.display_view = self
                        .display_texture
                        .as_ref()
                        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));

                    // Recreate gradient buffer with new dimensions
                    self.gradient_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Gradient Buffer"),
                        size: (trail_map_size * std::mem::size_of::<f32>()) as u64,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));

                    // Update bind group with new buffers and texture view
                    if let (
                        Some(agent_buffer),
                        Some(trail_map_buffer),
                        Some(gradient_buffer),
                        Some(sim_size_buffer),
                        Some(display_view),
                        Some(display_sampler),
                        Some(lut_buffer),
                        Some(pipeline_manager),
                    ) = (
                        &self.agent_buffer,
                        &self.trail_map_buffer,
                        &self.gradient_buffer,
                        &self.sim_size_buffer,
                        &self.display_view,
                        &self.display_sampler,
                        &self.lut_buffer,
                        &self.pipeline_manager,
                    ) {
                        self.bind_group_manager = Some(BindGroupManager::new(
                            device,
                            &pipeline_manager.compute_bind_group_layout,
                            &pipeline_manager.gradient_bind_group_layout,
                            &pipeline_manager.display_bind_group_layout,
                            &pipeline_manager.render_bind_group_layout,
                            agent_buffer,
                            trail_map_buffer,
                            gradient_buffer,
                            sim_size_buffer,
                            display_view,
                            display_sampler,
                            lut_buffer,
                        ));
                    }

                    // Update text renderer
                    if let Some(text_renderer) = &mut self.text_renderer {
                        text_renderer.update_window_size(physical_size.height);
                    }
                }
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.render();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl App {
    fn new() -> Self {
        let settings = Settings::default();
        let preset_manager = init_preset_manager();
        let preset_names = preset_manager.get_preset_names();
        let selected_preset = "Default".to_string();

        // Initialize LUT manager and get available LUTs
        let lut_manager = LutManager::new();
        let available_luts = lut_manager.get_available_luts();
        let current_lut_index = available_luts
            .iter()
            .position(|name| name == "MATPLOTLIB_bone_r")
            .expect("MATPLOTLIB_bone_r LUT not found");

        Self {
            adapter: None,
            agent_buffer: None,
            agent_count: 1_000_000,
            available_luts,
            bind_group_manager: None,
            config: None,
            current_lut_index,
            decay_rate_hi_range: false,
            device: None,
            display_sampler: None,
            display_texture: None,
            display_view: None,
            egui_renderer: None,
            frame_times: Vec::with_capacity(60),
            gradient_buffer: None,
            instance: None,
            last_frame_time: Instant::now(),
            lut_buffer: None,
            lut_manager,
            lut_preview_cache: HashMap::new(),
            lut_reversed: false,
            needs_display_update: false,
            needs_gpu_update: false,
            new_preset_name: String::new(),
            paused: false,
            pipeline_manager: None,
            preset_manager,
            preset_names,
            previous_agent_count: 1_000_000,
            previous_lut_index: current_lut_index,
            queue: None,
            save_preset_dialog_open: false,
            selected_preset,
            settings_changed: false,
            settings_have_changed: false,
            settings: settings.clone(),
            sim_size_buffer: None,
            surface: None,
            text_renderer: None,
            trail_map_buffer: None,
            ui_visible: true,
            window: None,
        }
    }

    /// Helper function to recreate bind groups when buffers change
    fn recreate_bind_groups(&mut self) {
        if let (
            Some(device),
            Some(agent_buffer),
            Some(trail_map_buffer),
            Some(gradient_buffer),
            Some(sim_size_buffer),
            Some(display_view),
            Some(display_sampler),
            Some(lut_buffer),
            Some(pipeline_manager),
        ) = (
            &self.device,
            &self.agent_buffer,
            &self.trail_map_buffer,
            &self.gradient_buffer,
            &self.sim_size_buffer,
            &self.display_view,
            &self.display_sampler,
            &self.lut_buffer,
            &self.pipeline_manager,
        ) {
            self.bind_group_manager = Some(BindGroupManager::new(
                device,
                &pipeline_manager.compute_bind_group_layout,
                &pipeline_manager.gradient_bind_group_layout,
                &pipeline_manager.display_bind_group_layout,
                &pipeline_manager.render_bind_group_layout,
                agent_buffer,
                trail_map_buffer,
                gradient_buffer,
                sim_size_buffer,
                display_view,
                display_sampler,
                lut_buffer,
            ));
        }
    }

    /// Helper function to handle agent count changes
    fn handle_agent_count_change(&mut self) {
        if self.agent_count != self.previous_agent_count {
            if let (Some(device), Some(config)) = (&self.device, &self.config) {
                self.agent_buffer = Some(create_agent_buffer(
                    device,
                    self.agent_count,
                    config.width,
                    config.height,
                    &self.settings,
                ));
                self.recreate_bind_groups();
                self.settings_changed = true;
                self.previous_agent_count = self.agent_count;
            }
        }
    }

    fn render(&mut self) {
        // Update FPS tracking
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;

        self.frame_times.push(frame_time);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }

        // Calculate average FPS over the last 60 frames
        let avg_frame_time: Duration =
            self.frame_times.iter().sum::<Duration>() / self.frame_times.len() as u32;

        // Update window title with FPS
        if let Some(window) = &self.window {
            window.set_title(&format!(
                "Physarum Simulation - {:.1} FPS",
                1.0 / avg_frame_time.as_secs_f64()
            ));
        }

        // Handle agent count changes immediately to prevent buffer overruns
        self.handle_agent_count_change();

        // Speed settings changed - first update uniform buffer, then reassign agent speeds
        if let (Some(sim_size_buffer), Some(queue), Some(config)) =
            (&self.sim_size_buffer, &self.queue, &self.config)
        {
            update_settings(
                &self.settings,
                sim_size_buffer,
                queue,
                config.width,
                config.height,
            );
        }

        // Now reassign existing agent speeds using GPU compute with updated uniform buffer
        if let (Some(device), Some(queue), Some(pipeline_manager), Some(bind_group_manager)) = (
            &self.device,
            &self.queue,
            &self.pipeline_manager,
            &self.bind_group_manager,
        ) {
            reassign_agent_speeds_gpu(
                device,
                queue,
                pipeline_manager,
                bind_group_manager,
                self.agent_count,
            );
        }

        if self.settings_changed
        {
            // Other settings changed - update uniform buffer
            if let (Some(sim_size_buffer), Some(queue), Some(config)) =
                (&self.sim_size_buffer, &self.queue, &self.config)
            {
                update_settings(
                    &self.settings,
                    sim_size_buffer,
                    queue,
                    config.width,
                    config.height,
                );
            }

            self.needs_display_update = true;
        }

        // Update LUT if it has changed
        if self.current_lut_index != self.previous_lut_index {
            if let (Some(queue), Some(lut_buffer)) = (&self.queue, &self.lut_buffer) {
                if let Ok(mut new_lut_data) = self
                    .lut_manager
                    .load_lut(&self.available_luts[self.current_lut_index])
                {
                    if self.lut_reversed {
                        new_lut_data.reverse();
                    }
                    let mut new_lut_data_combined = Vec::with_capacity(768);
                    new_lut_data_combined.extend_from_slice(&new_lut_data.red);
                    new_lut_data_combined.extend_from_slice(&new_lut_data.green);
                    new_lut_data_combined.extend_from_slice(&new_lut_data.blue);
                    let new_lut_data_u32: Vec<u32> =
                        new_lut_data_combined.iter().map(|&x| x as u32).collect();
                    queue.write_buffer(lut_buffer, 0, bytemuck::cast_slice(&new_lut_data_u32));
                    self.previous_lut_index = self.current_lut_index;
                }
            }
        }

        // Full rendering with simulation and UI
        if let (Some(surface), Some(device), Some(queue), Some(config)) =
            (&self.surface, &self.device, &self.queue, &self.config)
        {
            if let Ok(frame) = surface.get_current_texture() {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

                // Run compute passes for simulation
                if let (Some(pipeline_manager), Some(bind_group_manager)) =
                    (&self.pipeline_manager, &self.bind_group_manager)
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Simulation Compute Pass"),
                        timestamp_writes: None,
                    });

                    // Generate gradient (always update when settings change or at startup)
                    if self.settings.gradient_enabled
                        && (self.settings_changed
                            || self.needs_gpu_update
                            || self.needs_display_update)
                    {
                        cpass.set_pipeline(&pipeline_manager.gradient_pipeline);
                        cpass.set_bind_group(0, &bind_group_manager.gradient_bind_group, &[]);
                        cpass.dispatch_workgroups(
                            (config.width * config.height).div_ceil(256).min(65535),
                            1,
                            1,
                        );
                    }

                    // Only run simulation updates when not paused
                    if !self.paused {
                        // Update agent positions
                        cpass.set_pipeline(&pipeline_manager.compute_pipeline);
                        cpass.set_bind_group(0, &bind_group_manager.compute_bind_group, &[]);
                        cpass.dispatch_workgroups(
                            (self.agent_count as u32).div_ceil(256).min(65535),
                            1,
                            1,
                        );

                        // Decay trail map
                        cpass.set_pipeline(&pipeline_manager.decay_pipeline);
                        cpass.set_bind_group(0, &bind_group_manager.compute_bind_group, &[]);
                        cpass.dispatch_workgroups(
                            (config.width * config.height).div_ceil(256).min(65535),
                            1,
                            1,
                        );

                        // Diffuse trail map
                        cpass.set_pipeline(&pipeline_manager.diffuse_pipeline);
                        cpass.set_bind_group(0, &bind_group_manager.compute_bind_group, &[]);
                        cpass.dispatch_workgroups(
                            (config.width * config.height).div_ceil(256).min(65535),
                            1,
                            1,
                        );
                    }

                    // Always update display (even when paused) to show current state
                    if !self.paused || self.needs_display_update {
                        cpass.set_pipeline(&pipeline_manager.display_pipeline);
                        cpass.set_bind_group(0, &bind_group_manager.display_bind_group, &[]);
                        cpass.dispatch_workgroups(
                            config.width.div_ceil(16).min(65535),
                            config.height.div_ceil(16).min(65535),
                            1,
                        );
                        self.needs_display_update = false;
                    }
                }

                // Clear the framebuffer first
                {
                    let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Clear Pass"),
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
                }

                // Begin egui frame and draw UI
                // Track changes during UI processing
                let mut agent_count_changed = false;
                let mut new_agent_count = self.agent_count;
                let mut preset_to_apply: Option<String> = None;

                if let (Some(window), Some(egui_renderer)) = (&self.window, &mut self.egui_renderer)
                {
                    egui_renderer.begin_frame(window);

                    let full_output = egui_renderer.run_ui(window, |ctx| {
                        if self.ui_visible {
                            egui::SidePanel::left("settings_panel")
                                .resizable(true)
                                .default_width(300.0)
                                .show(ctx, |ui| {
                                    ui.heading("Simulation Settings");
                                    
                                    // Add keyboard shortcut note
                                    ui.label(egui::RichText::new("Press / (forward slash) to show/hide this panel").italics().color(egui::Color32::GRAY));
                                    ui.label(egui::RichText::new("Hover over an option to see a tooltip explaining it.").italics().color(egui::Color32::GRAY));
                                    ui.separator();
                                    
                                    // Add FPS display at the top
                                    ui.horizontal(|ui| {
                                        ui.label("FPS:");
                                        ui.label(format!("{:.1}", 1.0 / avg_frame_time.as_secs_f64()));
                                    });
                                    ui.separator();
                                    
                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                        // Presets
                                        ui.heading("Presets");
                                                                ui.horizontal(|ui| {
                            if ui.button("◀").clicked() {
                                let current_index = self.preset_names.iter().position(|name| name == &self.selected_preset).unwrap_or(0);
                                let prev_index = if current_index == 0 {
                                    self.preset_names.len() - 1
                                } else {
                                    current_index - 1
                                };
                                preset_to_apply = Some(self.preset_names[prev_index].clone());
                            }
                            egui::ComboBox::from_id_salt("preset_selector")
                                .selected_text(if self.settings_changed { "(unsaved)" } else { &self.selected_preset })
                                .show_ui(ui, |ui| {
                                    for name in &self.preset_names {
                                        if ui.selectable_label(&self.selected_preset == name, name).clicked() {
                                            preset_to_apply = Some(name.clone());
                                        }
                                    }
                                });
                            if ui.button("▶").clicked() {
                                let current_index = self.preset_names.iter().position(|name| name == &self.selected_preset).unwrap_or(0);
                                let next_index = (current_index + 1) % self.preset_names.len();
                                preset_to_apply = Some(self.preset_names[next_index].clone());
                            }
                        });
                                        
                                        // Save and Delete preset buttons
                                        ui.horizontal(|ui| {
                                            if ui.button("💾 Save Current").clicked() {
                                                self.save_preset_dialog_open = true;
                                                self.new_preset_name = String::new();
                                            }
                                            
                                            // Only show delete button for user presets (not built-in ones)
                                            let user_preset_names = self.preset_manager.get_user_preset_names();
                                            if user_preset_names.contains(&self.selected_preset) && ui.button("🗑 Delete").clicked() {
                                                if let Err(e) = self.preset_manager.delete_user_preset(&self.selected_preset) {
                                                    eprintln!("Failed to delete preset: {}", e);
                                                } else {
                                                    // Update the preset list after deletion
                                                    self.preset_names = self.preset_manager.get_preset_names();
                                                    
                                                    // Select default preset if current was deleted
                                                    if !self.preset_names.contains(&self.selected_preset) {
                                                        self.selected_preset = "Default".to_string();
                                                        // Immediately apply the default preset
                                                        if let Some(preset) = self.preset_manager.get_preset(&self.selected_preset) {
                                                            self.settings = preset.settings.clone();
                                                            self.needs_gpu_update = true;
                                                            self.settings_have_changed = false;  // Settings match preset (no "(unsaved)")
                                                            self.settings_changed = false;  // Not a manual settings change
                                                            
                                                            // Update the uniform buffer with new settings
                                                            if let (Some(sim_size_buffer), Some(queue), Some(config)) = (&self.sim_size_buffer, &self.queue, &self.config) {
                                                                update_settings(&self.settings, sim_size_buffer, queue, config.width, config.height);
                                                                                                                }
                                                    
                                                    // Reset trails and agents with new settings
                                                    if let (Some(trail_map_buffer), Some(agent_buffer), Some(queue), Some(config)) = 
                                                        (&self.trail_map_buffer, &self.agent_buffer, &self.queue, &self.config) {
                                                        reset_trails(trail_map_buffer, queue, config.width, config.height);
                                                        reset_agents(agent_buffer, queue, config.width, config.height, &self.settings, self.agent_count);
                                                        self.needs_display_update = true;
                                                    }
                                                }
                                            }
                                                }
                                            }
                                            
                                            // Refresh button to reload presets from files
                                            if ui.button("🔄 Refresh").clicked() {
                                                // Reload presets from filesystem
                                                self.preset_manager = slime_mold::presets::init_preset_manager();
                                                self.preset_names = self.preset_manager.get_preset_names();
                                                
                                                                                // Validate current selection still exists
                                if !self.preset_names.contains(&self.selected_preset) {
                                    preset_to_apply = Some("Default".to_string());
                                }
                                            }
                                        });
                                        
                                        // Save preset dialog
                                        if self.save_preset_dialog_open {
                                            egui::Window::new("Save Preset")
                                                .collapsible(false)
                                                .resizable(false)
                                                .show(ctx, |ui| {
                                                    ui.label("Enter preset name:");
                                                    ui.text_edit_singleline(&mut self.new_preset_name);
                                                    ui.horizontal(|ui| {
                                                        if ui.button("Save").clicked() && !self.new_preset_name.trim().is_empty() {
                                                            if let Err(e) = self.preset_manager.save_user_preset(&self.new_preset_name, &self.settings) {
                                                                eprintln!("Failed to save preset: {}", e);
                                                            } else {
                                                                // Reload presets to include the new one
                                                                self.preset_manager = slime_mold::presets::init_preset_manager();
                                                                self.preset_names = self.preset_manager.get_preset_names();
                                                                self.selected_preset = self.new_preset_name.clone();
                                                            }
                                                            self.save_preset_dialog_open = false;
                                                        }
                                                        if ui.button("Cancel").clicked() {
                                                            self.save_preset_dialog_open = false;
                                                        }
                                                    });
                                                });
                                        }
                                        
                                        ui.separator();

                                        // Color Scheme
                                        ui.heading("Color Scheme");
                                        ui.horizontal(|ui| {
                                            if ui.button("◀").clicked() {
                                                if self.current_lut_index > 0 {
                                                    self.current_lut_index -= 1;
                                                } else {
                                                    self.current_lut_index = self.available_luts.len() - 1;
                                                }
                                            }
                                            egui::ComboBox::from_id_salt("lut_selector")
                                                .selected_text(format!("{}{}", self.available_luts[self.current_lut_index], if self.lut_reversed { " (Reversed)" } else { "" }))
                                                .show_ui(ui, |ui| {
                                                    for (i, lut_name) in self.available_luts.iter().enumerate() {
                                                        ui.horizontal(|ui| {
                                                            // Use cache for LUT preview
                                                            let cache_key = (lut_name.clone(), self.lut_reversed);
                                                            let preview = self.lut_preview_cache.entry(cache_key.clone()).or_insert_with(|| {
                                                                if let Ok(mut lut_data) = self.lut_manager.load_lut(lut_name) {
                                                                    if self.lut_reversed {
                                                                        lut_data.reverse();
                                                                    }
                                                                    // Generate a Vec<egui::Color32> for the preview gradient
                                                                    (0..256).map(|idx| {
                                                                        egui::Color32::from_rgb(
                                                                            lut_data.red[idx],
                                                                            lut_data.green[idx],
                                                                            lut_data.blue[idx],
                                                                        )
                                                                    }).collect::<Vec<_>>()
                                                                } else {
                                                                    // Fallback: gray gradient
                                                                    (0..256).map(|idx| egui::Color32::from_gray(idx as u8)).collect::<Vec<_>>()
                                                                }
                                                            });
                                                            // Draw the gradient preview using the cached Vec<egui::Color32>
                                                            let rect = ui.allocate_rect(
                                                                egui::Rect::from_min_size(
                                                                    ui.min_rect().min,
                                                                    egui::vec2(50.0, ui.spacing().interact_size.y),
                                                                ),
                                                                egui::Sense::hover(),
                                                            );
                                                            let painter = ui.painter();
                                                            let rect = rect.rect;
                                                            let width = rect.width();
                                                            let steps = 50; // Number of gradient steps
                                                            let step_width = width / steps as f32;
                                                            for step in 0..steps {
                                                                let x = rect.min.x + step as f32 * step_width;
                                                                let t = step as f32 / steps as f32;
                                                                let idx = (t * 255.0) as usize;
                                                                let color = preview[idx];
                                                                painter.rect_filled(
                                                                    egui::Rect::from_min_size(
                                                                        egui::pos2(x, rect.min.y),
                                                                        egui::vec2(step_width, rect.height()),
                                                                    ),
                                                                    0.0,
                                                                    color,
                                                                );
                                                            }
                                                            ui.add_space(5.0);
                                                            // Add the LUT name
                                                            if ui.selectable_value(&mut self.current_lut_index, i, lut_name).clicked() {
                                                                ui.close_menu();
                                                            }
                                                        });
                                                    }
                                                });
                                            if ui.button("▶").clicked() {
                                                self.current_lut_index = (self.current_lut_index + 1) % self.available_luts.len();
                                            }
                                        });
                                        if ui.button("Reverse LUT").clicked() {
                                            self.lut_reversed = !self.lut_reversed;
                                            // Force LUT reload
                                            self.previous_lut_index = usize::MAX;
                                        }
                                        ui.separator();

                                        // Controls
                                        ui.heading("Controls");
                                        ui.horizontal(|ui| {
                                            // Pause/Resume button
                                            let pause_button_text = if self.paused { "▶ Resume" } else { "⏸ Pause" };
                                            if ui.button(pause_button_text).clicked() {
                                                self.paused = !self.paused;
                                            }
                                            
                                            if ui.button("Reset Trails").clicked() {
                                                if let (Some(trail_map_buffer), Some(queue), Some(config)) = (&self.trail_map_buffer, &self.queue, &self.config) {
                                                    reset_trails(trail_map_buffer, queue, config.width, config.height);
                                                    self.needs_display_update = true;
                                                }
                                            }
                                            if ui.button("Reset Agents").clicked() {
                                                if let (Some(agent_buffer), Some(queue), Some(config)) = (&self.agent_buffer, &self.queue, &self.config) {
                                                    reset_agents(agent_buffer, queue, config.width, config.height, &self.settings, self.agent_count);
                                                }
                                            }
                                        });
                                        if ui.button("🎲 Randomize Settings").clicked() {
                                            self.agent_count = randomize_settings(&mut self.settings, self.agent_count);
                                            
                                            // Mark settings as changed
                                            self.settings_changed = true;
                                            self.needs_display_update = true;
                                            
                                            // Check if settings still match current preset
                                            if self.preset_manager.get_preset(&self.selected_preset).is_some() {
                                                self.settings_have_changed = true;
                                            }
                                        }
                                        ui.separator();

                                        // Pheromone Settings
                                        ui.heading("Pheromone Settings");
                                        
                                        egui::Grid::new("pheromone_grid")
                                            .num_columns(2)
                                            .spacing([40.0, 4.0])
                                            .striped(true)
                                            .show(ui, |ui| {
                                                // Decay Rate with fine controls
                                                ui.label("Decay Rate").on_hover_text("Controls how fast trails disappear. Increasing this is a great way to lighten a slime mold that's too dense. A normal value is 0.1% (1.0 internally).");
                                                ui.horizontal(|ui| {
                                                    // Convert internal value to percent for display
                                                    let mut decay_percent = self.settings.pheromone_decay_rate * 0.1; // 1.0 = 0.1%
                                                    if ui.checkbox(&mut self.decay_rate_hi_range, "Lo/Hi").changed() {
                                                        // When switching to lo range, cap at 1%
                                                        if !self.decay_rate_hi_range && decay_percent > 1.0 {
                                                            decay_percent = 1.0;
                                                            self.settings.pheromone_decay_rate = decay_percent / 0.1;
                                                        }
                                                    }
                                                    if ui.button("−").clicked() {
                                                        if self.decay_rate_hi_range {
                                                            decay_percent = (decay_percent - 0.1).max(0.0);
                                                        } else {
                                                            decay_percent = (decay_percent - 0.01).max(0.0);
                                                        }
                                                        self.settings.pheromone_decay_rate = decay_percent / 0.1;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut decay_percent)
                                                        .range(0.0..=if self.decay_rate_hi_range { 100.0 } else { 10.0 })
                                                        .speed(if self.decay_rate_hi_range { 0.1 } else { 0.01 })
                                                        .suffix("%")
                                                    ).changed() {
                                                        self.settings.pheromone_decay_rate = decay_percent / 0.1;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        if self.decay_rate_hi_range {
                                                            decay_percent = (decay_percent + 0.1).min(10.0);
                                                        } else {
                                                            decay_percent = (decay_percent + 0.01).min(1.0);
                                                        }
                                                        self.settings.pheromone_decay_rate = decay_percent / 0.1;
                                                    }
                                                });
                                                ui.end_row();
                                                
                                                // Deposition Rate with fine controls
                                                ui.label("Deposition Rate").on_hover_text("At 0%, agents will not deposit any pheromones. At 100%, agents will saturate their location with the maximum amount of pheromones.");
                                                ui.horizontal(|ui| {
                                                    let mut deposition_percent = self.settings.pheromone_deposition_rate * 100.0;
                                                    if ui.button("−").clicked() {
                                                        deposition_percent = (deposition_percent - 1.0).max(0.0);
                                                        self.settings.pheromone_deposition_rate = deposition_percent / 100.0;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut deposition_percent).range(0.0..=100.0).speed(1.0).suffix("%")).changed() {
                                                        self.settings.pheromone_deposition_rate = deposition_percent / 100.0;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        deposition_percent = (deposition_percent + 1.0).min(100.0);
                                                        self.settings.pheromone_deposition_rate = deposition_percent / 100.0;
                                                    }
                                                });
                                                ui.end_row();
                                                
                                                // Diffusion Rate with fine controls
                                                ui.label("Diffusion Rate").on_hover_text("At 0%, pheromones will stay exactly where agents deposit them. At 100%, pheromones will spread to neighboring cells and dissapate.");
                                                ui.horizontal(|ui| {
                                                    let mut diffusion_percent = self.settings.pheromone_diffusion_rate * 100.0;
                                                    if ui.button("−").clicked() {
                                                        diffusion_percent = (diffusion_percent - 1.0).max(0.0);
                                                        self.settings.pheromone_diffusion_rate = diffusion_percent / 100.0;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut diffusion_percent).range(0.0..=100.0).speed(1.0).suffix("%")).changed() {
                                                        self.settings.pheromone_diffusion_rate = diffusion_percent / 100.0;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        diffusion_percent = (diffusion_percent + 1.0).min(100.0);
                                                        self.settings.pheromone_diffusion_rate = diffusion_percent / 100.0;
                                                    }
                                                });
                                                ui.end_row();
                                            });
                                        ui.separator();

                                        // Agent Settings
                                        ui.heading("Agent Settings");
                                        
                                        egui::Grid::new("agent_grid")
                                            .num_columns(2)
                                            .spacing([40.0, 4.0])
                                            .striped(true)
                                            .show(ui, |ui| {
                                                // Agent Count with buttons and number display
                                                ui.label("Agent Count").on_hover_text("Number of agents in the simulation. More agents create denser patterns but require more processing power.");
                                                ui.horizontal(|ui| {
                                                    let mut agent_count_m = (self.agent_count as f32 / 1_000_000.0).round();
                                                    if ui.button("−").clicked() {
                                                        agent_count_m = (agent_count_m - 1.0).max(0.0);
                                                        new_agent_count = (agent_count_m * 1_000_000.0) as usize;
                                                        agent_count_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut agent_count_m).range(0.0..=100.0).speed(0.1).suffix("M")).changed() {
                                                        new_agent_count = (agent_count_m * 1_000_000.0) as usize;
                                                        agent_count_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        agent_count_m = (agent_count_m + 1.0).min(100.0);
                                                        new_agent_count = (agent_count_m * 1_000_000.0) as usize;
                                                        agent_count_changed = true;
                                                    }
                                                });
                                                ui.end_row();
                                                
                                                // Min Speed with fine controls
                                                ui.label("Min Speed").on_hover_text("Minimum speed of agents. Lower values create more detailed patterns but slower movement.");
                                                ui.horizontal(|ui| {
                                                    if ui.button("−").clicked() {
                                                        self.settings.agent_speed_min = (self.settings.agent_speed_min - 0.1).max(0.0);
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut self.settings.agent_speed_min).range(0.0..=500.0).speed(1.0)).changed() {
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        self.settings.agent_speed_min = (self.settings.agent_speed_min + 0.1).min(self.settings.agent_speed_max);
                                                        self.settings_changed = true;
                                                    }
                                                });
                                                ui.end_row();
                                                
                                                // Max Speed with fine controls
                                                ui.label("Max Speed").on_hover_text("Maximum speed of agents. Higher values create more dynamic patterns but may be less stable.");
                                                ui.horizontal(|ui| {
                                                    if ui.button("−").clicked() {
                                                        self.settings.agent_speed_max = (self.settings.agent_speed_max - 0.1).max(self.settings.agent_speed_min);
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut self.settings.agent_speed_max).range(0.0..=500.0).speed(1.0)).changed() {
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        self.settings.agent_speed_max = (self.settings.agent_speed_max + 0.1).min(500.0);
                                                        self.settings_changed = true;
                                                    }
                                                });
                                                ui.end_row();
                                                
                                                // Turn Rate with fine controls (convert radians to degrees for display)
                                                ui.label("Turn Rate (deg/s)").on_hover_text("How quickly agents can change direction. Higher values create more dynamic, less predictable patterns.");
                                                ui.horizontal(|ui| {
                                                    let mut turn_rate_degrees = self.settings.agent_turn_rate * 180.0 / std::f32::consts::PI;
                                                    if ui.button("−").clicked() {
                                                        turn_rate_degrees = (turn_rate_degrees - 1.0).max(0.0);
                                                        self.settings.agent_turn_rate = turn_rate_degrees * std::f32::consts::PI / 180.0;
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut turn_rate_degrees).range(0.0..=360.0).speed(1.0).suffix(" deg/s")).changed() {
                                                        self.settings.agent_turn_rate = turn_rate_degrees * std::f32::consts::PI / 180.0;
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        turn_rate_degrees = (turn_rate_degrees + 1.0).min(360.0);
                                                        self.settings.agent_turn_rate = turn_rate_degrees * std::f32::consts::PI / 180.0;
                                                        self.settings_changed = true;
                                                    }
                                                });
                                                ui.end_row();
                                                
                                                // Jitter with fine controls
                                                ui.label("Jitter").on_hover_text("Random movement added to agent direction. Higher values create more chaotic, less organized patterns.");
                                                ui.horizontal(|ui| {
                                                    if ui.button("−").clicked() {
                                                        self.settings.agent_jitter = (self.settings.agent_jitter - 0.001).max(0.0);
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut self.settings.agent_jitter).range(0.0..=5.0).speed(0.001)).changed() {
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        self.settings.agent_jitter = (self.settings.agent_jitter + 0.001).min(5.0);
                                                        self.settings_changed = true;
                                                    }
                                                });
                                                ui.end_row();

                                                // Sensor Angle with fine controls (convert radians to degrees for display)
                                                ui.label("Sensor Angle (degrees)").on_hover_text("How wide the agent's sensor field is. Wider angles create more complex, branching patterns.");
                                                ui.horizontal(|ui| {
                                                    let mut sensor_angle_degrees = self.settings.agent_sensor_angle * 180.0 / std::f32::consts::PI;
                                                    if ui.button("−").clicked() {
                                                        sensor_angle_degrees = (sensor_angle_degrees - 0.5).max(0.0);
                                                        self.settings.agent_sensor_angle = sensor_angle_degrees * std::f32::consts::PI / 180.0;
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut sensor_angle_degrees).range(0.0..=180.0).speed(0.5).suffix(" deg")).changed() {
                                                        self.settings.agent_sensor_angle = sensor_angle_degrees * std::f32::consts::PI / 180.0;
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        sensor_angle_degrees = (sensor_angle_degrees + 0.5).min(180.0);
                                                        self.settings.agent_sensor_angle = sensor_angle_degrees * std::f32::consts::PI / 180.0;
                                                        self.settings_changed = true;
                                                    }
                                                });
                                                ui.end_row();
                                                
                                                // Sensor Distance with fine controls
                                                ui.label("Sensor Distance").on_hover_text("How far ahead agents can sense pheromones. Longer distances create more organized, network-like patterns.");
                                                ui.horizontal(|ui| {
                                                    if ui.button("−").clicked() {
                                                        self.settings.agent_sensor_distance = (self.settings.agent_sensor_distance - 1.0).max(0.0);
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut self.settings.agent_sensor_distance).range(0.0..=500.0).speed(1.0)).changed() {
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        self.settings.agent_sensor_distance = (self.settings.agent_sensor_distance + 1.0).min(500.0);
                                                        self.settings_changed = true;
                                                    }
                                                });
                                                ui.end_row();
                                            });
                                        
                                        // Ensure min speed doesn't exceed max speed
                                        if self.settings.agent_speed_min > self.settings.agent_speed_max {
                                            self.settings.agent_speed_max = self.settings.agent_speed_min;
                                        }
                                        if self.settings.agent_speed_max < self.settings.agent_speed_min {
                                            self.settings.agent_speed_min = self.settings.agent_speed_max;
                                        }

                                        // Starting Direction Range
                                        ui.heading("Starting Direction Range");
                                        let mut start_angle = self.settings.agent_possible_starting_headings.start;
                                        let mut end_angle = self.settings.agent_possible_starting_headings.end;
                                        
                                        egui::Grid::new("direction_grid")
                                            .num_columns(2)
                                            .spacing([40.0, 4.0])
                                            .striped(true)
                                            .show(ui, |ui| {
                                                // Start Angle with fine controls
                                                ui.label("Min Angle (degrees)");
                                                ui.horizontal(|ui| {
                                                    if ui.button("−").clicked() {
                                                        start_angle = (start_angle - 1.0).max(0.0);
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut start_angle).range(0.0..=360.0).speed(1.0)).changed() {
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        start_angle = (start_angle + 1.0).min(end_angle);
                                                        self.settings_changed = true;
                                                    }
                                                });
                                                ui.end_row();
                                                
                                                // End Angle with fine controls
                                                ui.label("Max Angle (degrees)");
                                                ui.horizontal(|ui| {
                                                    if ui.button("−").clicked() {
                                                        end_angle = (end_angle - 1.0).max(start_angle);
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.add(egui::DragValue::new(&mut end_angle).range(0.0..=360.0).speed(1.0)).changed() {
                                                        self.settings_changed = true;
                                                    }
                                                    if ui.button("+").clicked() {
                                                        end_angle = (end_angle + 1.0).min(360.0);
                                                        self.settings_changed = true;
                                                    }
                                                });
                                                ui.end_row();
                                            });
                                        
                                        if start_angle != self.settings.agent_possible_starting_headings.start || end_angle != self.settings.agent_possible_starting_headings.end {
                                            self.settings.agent_possible_starting_headings = start_angle.min(end_angle)..start_angle.max(end_angle);
                                        }
                                        ui.separator();

                                        // Gradient Settings
                                        ui.heading("Gradient Settings");
                                        
                                        egui::Grid::new("gradient_grid")
                                            .num_columns(2)
                                            .spacing([40.0, 4.0])
                                            .striped(true)
                                            .show(ui, |ui| {
                                                // Gradient Enabled
                                                ui.label("Enable Gradients").on_hover_text("Adds a constant pheromone gradient to influence agent movement. Can create interesting directional patterns.");
                                                if ui.checkbox(&mut self.settings.gradient_enabled, "").changed() {
                                                    self.settings_changed = true;
                                                }
                                                ui.end_row();
                                                
                                                if self.settings.gradient_enabled {
                                                    // Gradient Type
                                                    ui.label("Gradient Type").on_hover_text("Different gradient patterns that influence agent movement. Each creates unique emergent behaviors.");
                                                    egui::ComboBox::from_id_salt("gradient_type")
                                                        .selected_text(self.settings.gradient_type.as_str())
                                                        .show_ui(ui, |ui| {
                                                            for &gradient_type in slime_mold::settings::GradientType::all() {
                                                                if ui.selectable_value(&mut self.settings.gradient_type, gradient_type, gradient_type.as_str()).changed() {
                                                                    self.settings_changed = true;
                                                                }
                                                            }
                                                        });
                                                    ui.end_row();

                                                    // Gradient Strength
                                                    ui.label("Strength").on_hover_text("How strongly the gradient influences agent movement. Higher values create more pronounced directional patterns.");
                                                    ui.horizontal(|ui| {
                                                        if ui.button("−").clicked() {
                                                            self.settings.gradient_strength = (self.settings.gradient_strength - 1.0).max(0.0);
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.add(egui::DragValue::new(&mut self.settings.gradient_strength).range(0.0..=100.0).speed(1.0)).changed() {
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.button("+").clicked() {
                                                            self.settings.gradient_strength = (self.settings.gradient_strength + 1.0).min(100.0);
                                                            self.settings_changed = true;
                                                        }
                                                    });
                                                    ui.end_row();

                                                    // Center X
                                                    ui.label("Center X").on_hover_text("Horizontal position of the gradient center (0-100%). Affects where the gradient pattern is centered.");
                                                    ui.horizontal(|ui| {
                                                        let mut center_x_percent = self.settings.gradient_center_x * 100.0;
                                                        if ui.button("−").clicked() {
                                                            center_x_percent = (center_x_percent - 5.0).max(0.0);
                                                            self.settings.gradient_center_x = center_x_percent / 100.0;
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.add(egui::DragValue::new(&mut center_x_percent).range(0.0..=100.0).speed(1.0).suffix("%")).changed() {
                                                            self.settings.gradient_center_x = center_x_percent / 100.0;
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.button("+").clicked() {
                                                            center_x_percent = (center_x_percent + 5.0).min(100.0);
                                                            self.settings.gradient_center_x = center_x_percent / 100.0;
                                                            self.settings_changed = true;
                                                        }
                                                    });
                                                    ui.end_row();

                                                    // Center Y
                                                    ui.label("Center Y").on_hover_text("Vertical position of the gradient center (0-100%). Affects where the gradient pattern is centered.");
                                                    ui.horizontal(|ui| {
                                                        let mut center_y_percent = self.settings.gradient_center_y * 100.0;
                                                        if ui.button("−").clicked() {
                                                            center_y_percent = (center_y_percent - 5.0).max(0.0);
                                                            self.settings.gradient_center_y = center_y_percent / 100.0;
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.add(egui::DragValue::new(&mut center_y_percent).range(0.0..=100.0).speed(1.0).suffix("%")).changed() {
                                                            self.settings.gradient_center_y = center_y_percent / 100.0;
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.button("+").clicked() {
                                                            center_y_percent = (center_y_percent + 5.0).min(100.0);
                                                            self.settings.gradient_center_y = center_y_percent / 100.0;
                                                            self.settings_changed = true;
                                                        }
                                                    });
                                                    ui.end_row();

                                                    // Size (controls scale for all gradient types)
                                                    ui.label("Size").on_hover_text("Controls the scale of the gradient pattern. Larger values create more spread-out effects.");
                                                    ui.horizontal(|ui| {
                                                        if ui.button("−").clicked() {
                                                            self.settings.gradient_size = (self.settings.gradient_size - 0.05).max(0.1);
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.add(egui::DragValue::new(&mut self.settings.gradient_size).range(0.1..=2.0).speed(0.01)).changed() {
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.button("+").clicked() {
                                                            self.settings.gradient_size = (self.settings.gradient_size + 0.05).min(2.0);
                                                            self.settings_changed = true;
                                                        }
                                                    });
                                                    ui.end_row();

                                                    // Angle (rotates all gradient types)
                                                    ui.label("Angle (degrees)").on_hover_text("Rotates the gradient pattern. Affects the direction of influence on agent movement.");
                                                    ui.horizontal(|ui| {
                                                        if ui.button("−").clicked() {
                                                            self.settings.gradient_angle = (self.settings.gradient_angle - 5.0) % 360.0;
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.add(egui::DragValue::new(&mut self.settings.gradient_angle).range(0.0..=360.0).speed(1.0)).changed() {
                                                            self.settings_changed = true;
                                                        }
                                                        if ui.button("+").clicked() {
                                                            self.settings.gradient_angle = (self.settings.gradient_angle + 5.0) % 360.0;
                                                            self.settings_changed = true;
                                                        }
                                                    });
                                                    ui.end_row();
                                                }
                                            });
                                        ui.separator();
                                    });
                                });
                        }
                    });

                    // Render simulation to screen
                    if let (Some(pipeline_manager), Some(bind_group_manager)) =
                        (&self.pipeline_manager, &self.bind_group_manager)
                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Simulation Render Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });

                        // Render the simulation texture
                        rpass.set_pipeline(&pipeline_manager.render_pipeline);
                        rpass.set_bind_group(0, &bind_group_manager.render_bind_group, &[]);
                        rpass.draw(0..6, 0..1);
                    }

                    // End egui frame and draw with proper blending
                    use egui_wgpu::ScreenDescriptor;
                    let screen_descriptor = ScreenDescriptor {
                        size_in_pixels: [config.width, config.height],
                        pixels_per_point: window.scale_factor() as f32,
                    };

                    egui_renderer.end_frame_and_draw(
                        device,
                        queue,
                        &mut encoder,
                        window,
                        &view,
                        screen_descriptor,
                        full_output,
                    );
                }

                // Apply preset changes after UI processing
                if let Some(preset_name) = preset_to_apply {
                    if let Some(preset) = self.preset_manager.get_preset(&preset_name) {
                        self.settings = preset.settings.clone();
                        self.needs_gpu_update = true;
                        self.settings_have_changed = false;
                        self.settings_changed = false;
                        
                        // Update the uniform buffer with new settings
                        if let (Some(sim_size_buffer), Some(queue), Some(config)) = 
                            (&self.sim_size_buffer, &self.queue, &self.config) {
                            update_settings(&self.settings, sim_size_buffer, queue, config.width, config.height);
                        }
                        
                        // Reset trails and agents with new settings
                        if let (Some(trail_map_buffer), Some(agent_buffer), Some(queue), Some(config)) = 
                            (&self.trail_map_buffer, &self.agent_buffer, &self.queue, &self.config) {
                            reset_trails(trail_map_buffer, queue, config.width, config.height);
                            reset_agents(agent_buffer, queue, config.width, config.height, &self.settings, self.agent_count);
                            self.needs_display_update = true;
                        }
                    }
                    self.selected_preset = preset_name;
                }

                // Handle agent count changes after UI processing (outside egui scope)
                if agent_count_changed {
                    self.agent_count = new_agent_count;
                    if let (Some(device), Some(config)) = (&self.device, &self.config) {
                        self.agent_buffer = Some(create_agent_buffer(
                            device,
                            self.agent_count,
                            config.width,
                            config.height,
                            &self.settings,
                        ));
                        
                        // Recreate bind groups inline to avoid borrowing issues
                        if let (
                            Some(agent_buffer),
                            Some(trail_map_buffer),
                            Some(gradient_buffer),
                            Some(sim_size_buffer),
                            Some(display_view),
                            Some(display_sampler),
                            Some(lut_buffer),
                            Some(pipeline_manager),
                        ) = (
                            &self.agent_buffer,
                            &self.trail_map_buffer,
                            &self.gradient_buffer,
                            &self.sim_size_buffer,
                            &self.display_view,
                            &self.display_sampler,
                            &self.lut_buffer,
                            &self.pipeline_manager,
                        ) {
                            self.bind_group_manager = Some(BindGroupManager::new(
                                device,
                                &pipeline_manager.compute_bind_group_layout,
                                &pipeline_manager.gradient_bind_group_layout,
                                &pipeline_manager.display_bind_group_layout,
                                &pipeline_manager.render_bind_group_layout,
                                agent_buffer,
                                trail_map_buffer,
                                gradient_buffer,
                                sim_size_buffer,
                                display_view,
                                display_sampler,
                                lut_buffer,
                            ));
                        }
                        
                        self.settings_changed = true;
                        self.previous_agent_count = self.agent_count;
                    }
                }

                // Submit the command buffer
                queue.submit(std::iter::once(encoder.finish()));
                frame.present();
            }
        }
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
