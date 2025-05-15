use crate::{
    DEFAULT_SETTINGS_FILE,
    agent::{Agent, AgentUpdate},
    errors::SlimeError,
    pheromones::Pheromones,
    rect::Rect,
    settings::Settings,
};
use log::{error, info};
use rayon::prelude::*;
use std::fs;
use std::sync::{Arc, RwLock};

pub struct World {
    agents: Vec<Agent>,
    frame_time: f32,
    pheromones: Arc<RwLock<Pheromones>>,
    settings: Settings,
    boundary_rect: Rect<u32>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // New GPU resources for display texture conversion
    display_texture: wgpu::Texture,
    display_texture_view: wgpu::TextureView,
    pheromone_data_buffer: wgpu::Buffer,
    conversion_pipeline: wgpu::ComputePipeline,
    conversion_bind_group_layout: wgpu::BindGroupLayout,
    conversion_bind_group: wgpu::BindGroup,
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    pub fn new(settings: Settings, device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        info!("generating {} agents", settings.agent_count);
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
ENABLE_DYN_GRAD	{:?}
"#,
            settings.agent_count_maximum,
            settings.agent_jitter,
            settings.agent_speed_min,
            settings.agent_speed_max,
            settings.agent_turn_speed,
            settings.agent_possible_starting_headings,
            settings.agent_deposition_amount,
            settings.pheromone_decay_factor,
            settings.pheromone_enable_dynamic_gradient,
        );

        let agents: Vec<_> = (0..settings.agent_count)
            .map(|_| Agent::new_from_settings(&settings))
            .collect();

        let pheromones = Arc::new(RwLock::new(Pheromones::new(
            settings.window_width,
            settings.window_height,
            settings.pheromone_decay_factor,
            settings.pheromone_enable_dynamic_gradient,
            None,
            Arc::clone(&device),
            Arc::clone(&queue),
        )));

        let boundary_rect = Rect::new(0, 0, settings.window_width, settings.window_height);

        // --- Initialize new GPU resources for display ---
        let texture_size = wgpu::Extent3d {
            width: settings.window_width,
            height: settings.window_height,
            depth_or_array_layers: 1,
        };

        let display_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("World Display Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm, // Shader writes rgba8unorm
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let display_texture_view =
            display_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let buffer_size = (settings.window_width
            * settings.window_height
            * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;
        let pheromone_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pheromone Data Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load and create compute shader and pipeline
        let shader_source_str = fs::read_to_string("src/shaders/world_convert.wgsl")
            .expect("Failed to read world_convert.wgsl shader");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("World Convert Shader Module"),
            source: wgpu::ShaderSource::Wgsl(shader_source_str.into()),
        });

        let conversion_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("World Convert Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        // Pheromone values buffer
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None, // Buffer size checked at dispatch
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        // Output texture
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let conversion_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("World Convert Pipeline Layout"),
                bind_group_layouts: &[&conversion_bind_group_layout],
                push_constant_ranges: &[],
            });

        let conversion_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("World Conversion Compute Pipeline"),
                layout: Some(&conversion_pipeline_layout),
                module: &shader_module,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        let conversion_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("World Convert Bind Group"),
            layout: &conversion_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pheromone_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&display_texture_view),
                },
            ],
        });

        Self {
            agents,
            boundary_rect,
            frame_time: 0.0,
            pheromones,
            settings,
            device,
            queue,
            display_texture,
            display_texture_view,
            pheromone_data_buffer,
            conversion_pipeline,
            conversion_bind_group_layout,
            conversion_bind_group,
        }
    }

    pub fn window_width(&self) -> u32 {
        self.settings.window_width
    }

    pub fn window_height(&self) -> u32 {
        self.settings.window_height
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.settings.window_width = width;
        self.settings.window_height = height;
        self.boundary_rect = Rect::new(0, 0, width, height);

        self.pheromones = Arc::new(RwLock::new(Pheromones::new(
            width,
            height,
            self.settings.pheromone_decay_factor,
            self.settings.pheromone_enable_dynamic_gradient,
            None,
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
        )));
        info!("World resized pheromones to {}x{}", width, height);

        // --- Recreate GPU resources for display ---
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        self.display_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("World Display Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.display_texture_view = self
            .display_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let buffer_size =
            (width * height * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;
        self.pheromone_data_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pheromone Data Buffer (Resized)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Recreate bind group with new buffer and texture view
        self.conversion_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("World Convert Bind Group (Resized)"),
            layout: &self.conversion_bind_group_layout, // Layout is reused
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.pheromone_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.display_texture_view),
                },
            ],
        });
        info!("World display resources resized to {}x{}", width, height);
    }

    pub fn reload_settings(&mut self) -> Result<(), SlimeError> {
        let new_settings = Settings::load_from_file(DEFAULT_SETTINGS_FILE)?;
        let agent_settings_changed = self.settings.did_agent_settings_change(&new_settings);
        let pheromone_settings_changed = self.settings.did_pheromone_settings_change(&new_settings);
        self.settings = new_settings;

        if agent_settings_changed {
            info!("agent settings have changed, running an update...");
            if let Err(e) = self.refresh_agents_from_settings() {
                error!(
                    "failed to update agents after a settings change with error: {}",
                    e
                );
            }
        }

        if pheromone_settings_changed {
            info!("pheromone settings have changed, running an update...");
            if let Err(e) = self.refresh_pheromones_from_settings() {
                error!(
                    "failed to update pheromones after a settings change with error: {}",
                    e
                );
            }
        }

        if self.agents.len() < self.settings.agent_count {
            (self.agents.len()..self.settings.agent_count)
                .for_each(|_| self.agents.push(Agent::new_from_settings(&self.settings)));
        }

        Ok(())
    }

    pub fn set_frame_time(&mut self, frame_time: f32) {
        self.frame_time = frame_time;
    }

    pub fn toggle_dynamic_gradient(&mut self) {
        self.settings.pheromone_enable_dynamic_gradient =
            !self.settings.pheromone_enable_dynamic_gradient;

        let _ = self.refresh_pheromones_from_settings();
    }

    fn refresh_agents_from_settings(&mut self) -> Result<(), SlimeError> {
        let agent_update = AgentUpdate {
            jitter: Some(self.settings.agent_jitter),
            rotation_speed: Some(self.settings.agent_turn_speed),
            deposition_amount: Some(self.settings.agent_deposition_amount),
            ..Default::default()
        };

        let move_speed_range = self.settings.agent_speed_min..self.settings.agent_speed_max;

        self.agents.iter_mut().for_each(|agent| {
            agent.apply_update(&agent_update);
            agent.set_new_random_move_speed_in_range(move_speed_range.clone());
        });

        Ok(())
    }

    fn refresh_pheromones_from_settings(&mut self) -> Result<(), SlimeError> {
        let mut pheromones = self
            .pheromones
            .write()
            .map_err(|e| SlimeError::ThreadSafety(e.to_string()))?;

        if self.settings.pheromone_enable_dynamic_gradient {
            pheromones.enable_dynamic_gradient();
        } else {
            pheromones.disable_dynamic_gradient();
        }

        pheromones.set_decay_factor(self.settings.pheromone_decay_factor);

        Ok(())
    }

    /// Update the `World` internal state; bounce the box around the screen.
    pub fn update(&mut self) {
        let boundary_rect = &self.boundary_rect;
        let pheromones_read_lock = &self.pheromones;
        let agents = &mut self.agents;
        let delta_t = self.frame_time;

        // Agents sense the environment (using CPU cache of pheromones from end of *previous* frame)
        // and decide on their next action.
        agents.par_iter_mut().for_each(|agent| {
            let pheromones_guard = pheromones_read_lock
                .read()
                .expect("reading pheromones during agent update");
            // Agent::update now uses pheromones_guard.get_reading(), which reads from CPU cache
            agent.update(&pheromones_guard, delta_t, boundary_rect)
        });

        // Now, update the pheromone maps based on agent actions from this frame
        // and other pheromone dynamics (diffuse, decay).
        // These operations work on the GPU textures.
        let mut pheromones_write_guard = self
            .pheromones
            .write()
            .expect("couldn't get write lock on pheromones for update steps");

        // Deposit step (GPU)
        pheromones_write_guard.deposit(agents); // Pass current state of agents for deposition

        // Diffuse step (currently GPU placeholder - copies texture)
        pheromones_write_guard.diffuse();

        // Decay step (GPU)
        pheromones_write_guard.decay();

        // CRITICAL: After all GPU pheromone operations for the current frame are done,
        // update the CPU read cache with the latest pheromone state.
        // This cache will be used by agents in the *next* frame's sensing phase.
        pheromones_write_guard.update_cpu_read_cache();

        // Drop the write guard before calling update_display_texture,
        // as update_display_texture needs a read guard.
        drop(pheromones_write_guard);

        // New step: update our owned display texture using GPU
        self.update_display_texture();

        // Agent population control (remains the same)
        if self.agents.len() > self.settings.agent_count_maximum {
            self.agents.truncate(self.settings.agent_count_maximum)
        }
    }

    /// Updates the internal display_texture using a compute shader.
    /// This replaces the CPU-bound logic of the old `draw` method.
    pub fn update_display_texture(&self) {
        let pheromones_guard = self
            .pheromones
            .read()
            .expect("couldn't get lock on pheromones for display texture update");

        let pheromones_row_major = pheromones_guard.read_current_texture_to_vec();

        if pheromones_row_major.is_empty() {
            error!("Pheromone data is empty. Skipping display texture update.");
            return;
        }

        // Ensure buffer has correct size for the data.
        // This should be guaranteed by resize logic, but a check might be good for safety.
        let expected_buffer_elements = self.settings.window_width * self.settings.window_height;
        if pheromones_row_major.len() != expected_buffer_elements as usize {
            error!(
                "Pheromone data size mismatch for display update. Expected {}, got {}. Skipping update.",
                expected_buffer_elements,
                pheromones_row_major.len()
            );
            return;
        }

        // Write pheromone data to GPU buffer
        self.queue.write_buffer(
            &self.pheromone_data_buffer,
            0,
            bytemuck::cast_slice(&pheromones_row_major),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("World Display Texture Update Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("World Pheromone to RGBA Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.conversion_pipeline);
            compute_pass.set_bind_group(0, &self.conversion_bind_group, &[]);
            // Dispatch based on texture size; workgroup size is 8x8 in shader
            let workgroup_size_x = 8;
            let workgroup_size_y = 8;
            compute_pass.dispatch_workgroups(
                (self.settings.window_width + workgroup_size_x - 1) / workgroup_size_x, // Ceil division
                (self.settings.window_height + workgroup_size_y - 1) / workgroup_size_y, // Ceil division
                1,
            );
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Provides a view to the GPU-managed texture that holds the visual representation of pheromones.
    pub fn get_display_texture_view(&self) -> &wgpu::TextureView {
        &self.display_texture_view
    }
}
