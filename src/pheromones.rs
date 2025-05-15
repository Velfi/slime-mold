use crate::{Agent, Point2, Swapper};
use log::{debug, trace};
use nalgebra::DMatrix;
use pollster;
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub type StaticGradientGeneratorFn = Box<dyn Fn(u32, u32) -> Vec<f32>>;

// GPU compute shader for box blur (WGSL)
const BOX_BLUR_SHADER_WGSL: &str = include_str!("shaders/box_blur.wgsl");

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBlurUniforms {
    width: u32,
    height: u32,
    x_radius: u32,
    y_radius: u32,
}

async fn box_filter_gpu(
    image_data: &DMatrix<f32>,
    width: u32,
    height: u32,
    x_radius: u32,
    y_radius: u32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> DMatrix<f32> {
    // --- Initialization ---
    // let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    //     backends: wgpu::Backends::all(),
    //     ..Default::default()
    // });

    // let adapter_opt = instance
    //     .request_adapter(&wgpu::RequestAdapterOptions {
    //         power_preference: wgpu::PowerPreference::HighPerformance,
    //         compatible_surface: None,
    //         force_fallback_adapter: false,
    //     })
    //     .await;

    // let adapter = match adapter_opt {
    //     Some(a) => a,
    //     None => {
    //         log::warn!("Failed to find suitable GPU adapter. Falling back to CPU blur.");
    //         return box_filter(image_data, width, height, x_radius, y_radius);
    //     }
    // };

    // let (device, queue) = adapter
    //     .request_device(
    //         &wgpu::DeviceDescriptor {
    //             label: Some("Box Blur Device"),
    //             required_features: wgpu::Features::empty(),
    //             required_limits: wgpu::Limits::default(),
    //         },
    //         None,
    //     )
    //     .await
    //     .expect("Failed to create GPU device");

    // --- Shader Module ---
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Box Blur Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(BOX_BLUR_SHADER_WGSL)),
    });

    // --- Data Preparation ---
    // Flatten DMatrix (column-major) to Vec<f32> (row-major) for GPU
    let mut flat_input_data: Vec<f32> = Vec::with_capacity((width * height) as usize);
    for r in 0..image_data.nrows() {
        for c in 0..image_data.ncols() {
            flat_input_data.push(image_data[(r, c)]);
        }
    }

    let input_buffer_size =
        (flat_input_data.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let output_buffer_size = input_buffer_size;

    // --- Buffers ---
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Image Buffer"),
        contents: bytemuck::cast_slice(&flat_input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Image Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let uniforms = GpuBlurUniforms {
        width,
        height,
        x_radius,
        y_radius,
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // --- Bind Group & Pipeline ---
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Box Blur Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                // Input image
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // Output image
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                // Uniforms
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Box Blur Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Box Blur Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Box Blur Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    // --- Command Encoding & Submission ---
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Box Blur Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Box Blur Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Workgroup size is 8x8 (see shader)
        const WORKGROUP_SIZE_X: u32 = 8;
        const WORKGROUP_SIZE_Y: u32 = 8;
        let dispatch_width = (width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let dispatch_height = (height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
    }

    // --- Readback ---
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);

    queue.submit(std::iter::once(encoder.finish()));

    // --- Get Data from GPU ---
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel(); // Channel for async callback
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).expect("Failed to send map_async result");
    });
    device.poll(wgpu::Maintain::Wait); // Poll device to ensure callback is processed

    rx.recv()
        .expect("Failed to receive map_async result from channel")
        .expect("Failed to map staging buffer");

    let data = buffer_slice.get_mapped_range();
    let result_slice: &[f32] = bytemuck::cast_slice(&data);
    let result_vec: Vec<f32> = result_slice.to_vec();

    drop(data); // Explicitly drop mapped range before unmap
    staging_buffer.unmap();

    // Reconstruct DMatrix from row-major Vec<f32>
    DMatrix::from_row_slice(height as usize, width as usize, &result_vec)
}

async fn read_static_gradient_texel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    x: u32,
    y: u32,
) -> Option<f32> {
    let texture_format = texture.format(); // e.g., R32Float
    // Get bytes per pixel/block. For R32Float, this is 4 bytes.
    let bytes_per_pixel = match texture_format.block_copy_size(Some(wgpu::TextureAspect::All)) {
        Some(size) => size,
        None => {
            log::error!(
                "Could not determine block_copy_size for texture format {:?}",
                texture_format
            );
            return None;
        }
    };

    if bytes_per_pixel == 0 {
        log::error!(
            "Texture format {:?} has a block_copy_size of 0.",
            texture_format
        );
        return None;
    }

    let buffer_size = bytes_per_pixel as wgpu::BufferAddress;
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Static Gradient Readback Staging Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Static Gradient Readback Encoder"),
    });

    let copy_size = wgpu::Extent3d {
        width: 1, // We are copying a single texel
        height: 1,
        depth_or_array_layers: 1,
    };

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x, y, z: 0 }, // Coordinates of the texel to copy
            aspect: wgpu::TextureAspect::All,      // Assuming single-aspect format like R32Float
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                // For a single row (height=1), bytes_per_row can be None (tightly packed)
                // or Some(bytes_per_pixel * width_of_copy) if not already aligned for multi-row copies.
                // Since we copy width=1, height=1, tightly packed is fine.
                bytes_per_row: None,
                rows_per_image: None, // Since height is 1
            },
        },
        copy_size,
    );

    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        if tx.send(result).is_err() {
            log::error!("Failed to send map_async result: receiver dropped");
        }
    });

    // Poll the device to ensure the submission is processed and the callback is called.
    // This is a blocking call in an async function, which is generally okay if this
    // function itself is called via pollster::block_on from a sync context.
    device.poll(wgpu::Maintain::Wait);

    match rx.recv() {
        Ok(Ok(())) => {
            let value = {
                let data = buffer_slice.get_mapped_range();
                // Ensure we only read the expected number of bytes for one f32
                if data.len() >= std::mem::size_of::<f32>() {
                    bytemuck::from_bytes::<f32>(&data[..std::mem::size_of::<f32>()]).clone()
                } else {
                    log::error!("Mapped buffer too small for f32. Size: {}", data.len());
                    staging_buffer.unmap();
                    return None;
                }
            };
            // Important: drop the mapped range view *before* unmapping the buffer.
            // `get_mapped_range` returns a guard that must be dropped.
            // The previous line `let value = { ... }` ensures `data` is dropped here.
            staging_buffer.unmap();
            Some(value)
        }
        Ok(Err(e)) => {
            log::error!("Failed to map static gradient staging buffer: {:?}", e);
            // staging_buffer.unmap(); // Buffer is not mapped in this case, unmap would panic.
            None
        }
        Err(e) => {
            log::error!("Failed to receive map_async result from channel: {:?}", e);
            // Buffer state is uncertain, unmapping might be risky / no-op.
            None
        }
    }
}

pub struct Pheromones {
    grid: Swapper<DMatrix<f32>>,
    static_gradient_texture: Option<wgpu::Texture>,
    enable_static_gradient: bool,
    enable_dynamic_gradient: bool,
    decay_factor: f32,
    height: u32,
    width: u32,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl Pheromones {
    pub fn new(
        width: u32,
        height: u32,
        decay_factor: f32,
        enable_dynamic_gradient: bool,
        static_gradient_generator: Option<StaticGradientGeneratorFn>,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        let grid_matrix = DMatrix::zeros(height as usize, width as usize);
        let mut static_gradient_texture_opt = None;
        let mut enable_static_gradient_flag = false;

        if let Some(generator) = &static_gradient_generator {
            let sg_vec = generator(width, height);
            if sg_vec.len() == (width * height) as usize {
                let texture_size = wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                };
                let static_texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Static Gradient Texture"),
                    size: texture_size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC, // For potential readback
                    view_formats: &[],
                });

                queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &static_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&sg_vec),
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(std::mem::size_of::<f32>() as u32 * width),
                        rows_per_image: Some(height),
                    },
                    texture_size,
                );
                static_gradient_texture_opt = Some(static_texture);
                enable_static_gradient_flag = true;
            } else {
                debug!(
                    "Static gradient generator produced a Vec of incorrect size. Expected {}, got {}. Static gradient disabled.",
                    width * height,
                    sg_vec.len()
                );
            }
        }

        debug!(
            "Created new DMatrix grid with height {} and width {}",
            height, width
        );

        let grid = Swapper::new(grid_matrix.clone(), grid_matrix);

        Self {
            decay_factor,
            enable_static_gradient: enable_static_gradient_flag,
            enable_dynamic_gradient,
            grid,
            static_gradient_texture: static_gradient_texture_opt,
            height,
            width,
            device,
            queue,
        }
    }

    pub fn get_reading(&self, at_location: Point2) -> Option<i32> {
        let (x_f, y_f) = (at_location.x.round(), at_location.y.round());
        let is_within_bounds =
            x_f >= 0.0 && x_f < self.width as f32 && y_f >= 0.0 && y_f < self.height as f32;

        if is_within_bounds {
            let (x_u32, y_u32) = (x_f as u32, y_f as u32);
            let row = y_u32 as usize;
            let col = x_u32 as usize;

            match (self.enable_dynamic_gradient, self.enable_static_gradient) {
                (true, true) => {
                    // Dynamic and Static
                    let pheromone_value = self.grid.a()[(row, col)];
                    let static_value_f32_opt: Option<f32> =
                        if let Some(tex) = &self.static_gradient_texture {
                            pollster::block_on(read_static_gradient_texel(
                                &self.device,
                                &self.queue,
                                tex,
                                x_u32,
                                y_u32,
                            ))
                        } else {
                            // This case should ideally not happen if enable_static_gradient is true
                            // but the texture failed to initialize. Log if it does.
                            log::warn!("Static gradient enabled but no texture found for reading.");
                            None
                        };
                    // Default to 0.0 if texture read fails or no texture present
                    let static_value = static_value_f32_opt.unwrap_or(0.0);
                    Some(((static_value + pheromone_value) * 255.0).round() as i32)
                }
                (true, false) => Some((self.grid.a()[(row, col)] * 255.0).round() as i32),
                (false, true) => {
                    // Static only
                    let static_value_f32_opt: Option<f32> =
                        if let Some(tex) = &self.static_gradient_texture {
                            pollster::block_on(read_static_gradient_texel(
                                &self.device,
                                &self.queue,
                                tex,
                                x_u32,
                                y_u32,
                            ))
                        } else {
                            log::warn!("Static gradient enabled but no texture found for reading.");
                            None
                        };
                    // Default to an intensity of 0 if texture read fails or no texture
                    static_value_f32_opt
                        .map(|sv| (sv * 255.0).round() as i32)
                        .or(Some(0))
                }
                (false, false) => Some(0),
            }
        } else {
            None
        }
    }

    pub fn len(&self) -> u32 {
        self.width * self.height
    }

    pub fn deposit(&mut self, agents: &[Agent]) {
        if !self.enable_dynamic_gradient {
            return;
        }

        let (current_pheromones_data, next_pheromones_data) = self.grid.read_a_write_b();

        *next_pheromones_data = current_pheromones_data.clone();

        for agent in agents {
            let location_to_deposit: Point2 = agent.location();
            let x_f32 = location_to_deposit.x;
            let y_f32 = location_to_deposit.y;

            // Wrap coordinates
            let x_wrapped = (x_f32 % self.width as f32 + self.width as f32) % self.width as f32;
            let y_wrapped = (y_f32 % self.height as f32 + self.height as f32) % self.height as f32;

            let x = x_wrapped as u32;
            let y = y_wrapped as u32;
            let row = y as usize;
            let col = x as usize;

            if row < next_pheromones_data.nrows() && col < next_pheromones_data.ncols() {
                next_pheromones_data[(row, col)] = agent.deposition_amount();
            } else {
                trace!(
                    "Calculated index ({}, {}) out of bounds for DMatrix with shape ({}, {}) during deposition",
                    row,
                    col,
                    next_pheromones_data.nrows(),
                    next_pheromones_data.ncols()
                );
            }
        }
        self.grid.swap();
    }

    pub fn diffuse(&mut self) {
        if !self.enable_dynamic_gradient {
            return;
        }
        let (grid_a_data, grid_b_data) = self.grid.read_a_write_b();

        let filtered_matrix = pollster::block_on(box_filter_gpu(
            grid_a_data,
            self.width,
            self.height,
            1, // x_radius
            1, // y_radius
            &self.device,
            &self.queue,
        ));
        *grid_b_data = filtered_matrix;

        self.grid.swap()
    }

    pub fn decay(&mut self) {
        if !self.enable_dynamic_gradient {
            return;
        }

        let (current_pheromones_data, next_pheromones_data) = self.grid.read_a_write_b();
        let decay_factor = self.decay_factor;

        *next_pheromones_data = current_pheromones_data * (1.0 - decay_factor);

        self.grid.swap();
    }

    pub fn set_decay_factor(&mut self, decay_factor: f32) {
        self.decay_factor = decay_factor;
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.grid.a().iter()
    }

    pub fn enable_dynamic_gradient(&mut self) {
        self.enable_dynamic_gradient = true;
    }

    pub fn disable_dynamic_gradient(&mut self) {
        self.enable_dynamic_gradient = false;
    }

    pub fn get_current_grid(&self) -> &DMatrix<f32> {
        self.grid.a()
    }
}

#[allow(dead_code)] // Added to suppress unused function warning
fn box_filter(
    image_data: &DMatrix<f32>,
    width: u32,
    height: u32,
    x_radius: u32,
    y_radius: u32,
) -> DMatrix<f32> {
    let matrix_rows = image_data.nrows();
    let matrix_cols = image_data.ncols();

    let mut out_data = DMatrix::zeros(matrix_rows, matrix_cols);

    let kernel_width = 2 * x_radius + 1;
    let kernel_height = 2 * y_radius + 1;
    let kernel_size = (kernel_width * kernel_height) as f32;

    for y_out_u in 0..height {
        for x_out_u in 0..width {
            let y_out = y_out_u as usize;
            let x_out = x_out_u as usize;
            let mut sum = 0.0;
            for ky in 0..kernel_height {
                for kx in 0..kernel_width {
                    let x_in_intermediate = x_out_u as i32 + kx as i32 - x_radius as i32;
                    let y_in_intermediate = y_out_u as i32 + ky as i32 - y_radius as i32;

                    // Wrap coordinates
                    let x_in = (x_in_intermediate % width as i32 + width as i32) % width as i32;
                    let y_in = (y_in_intermediate % height as i32 + height as i32) % height as i32;

                    sum += image_data[(y_in as usize, x_in as usize)];
                }
            }
            let avg = sum / kernel_size;
            out_data[(y_out, x_out)] = avg;
        }
    }
    out_data
}
