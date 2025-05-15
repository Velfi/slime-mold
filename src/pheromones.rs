use crate::{Agent, Point2};
use log::debug;
use nalgebra::DMatrix;
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub type StaticGradientGeneratorFn = Box<dyn Fn(u32, u32) -> Vec<f32>>;

// GPU compute shader for decay (WGSL)
const DECAY_SHADER_WGSL: &str = include_str!("shaders/decay.wgsl");
// GPU compute shader for deposit (WGSL)
const DEPOSIT_SHADER_WGSL: &str = include_str!("shaders/deposit.wgsl");
// GPU compute shader for diffuse (WGSL)
const DIFFUSE_SHADER_WGSL: &str = include_str!("shaders/diffuse.wgsl");

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DecayUniforms {
    decay_factor: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuAgent {
    position: [f32; 2],
    deposition_amount: f32,
    _padding: f32, // Ensure 16-byte alignment for SSBO array elements
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DepositUniforms {
    texture_width: u32,
    texture_height: u32,
    num_agents: u32,
    _padding: u32, // Ensure 16-byte alignment for UBO
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DiffuseUniforms {
    texture_width: u32,
    texture_height: u32,
    _padding1: u32, // Match WGSL struct padding
    _padding2: u32,
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
                    *bytemuck::from_bytes::<f32>(&data[..std::mem::size_of::<f32>()])
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
    texture_a: wgpu::Texture,
    texture_b: wgpu::Texture,
    texture_a_view: wgpu::TextureView,
    texture_b_view: wgpu::TextureView,
    current_texture_is_a: bool,
    static_gradient_texture: Option<wgpu::Texture>,
    enable_static_gradient: bool,
    enable_dynamic_gradient: bool,
    decay_factor: f32,
    height: u32,
    width: u32,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Decay compute pipeline resources
    decay_pipeline: wgpu::ComputePipeline,
    decay_bind_group_layout: wgpu::BindGroupLayout,

    // Deposit compute pipeline resources
    deposit_pipeline: wgpu::ComputePipeline,
    deposit_bind_group_layout: wgpu::BindGroupLayout,

    // Diffuse compute pipeline resources
    diffuse_pipeline: wgpu::ComputePipeline,
    diffuse_bind_group_layout: wgpu::BindGroupLayout,

    // CPU cache for agent reading
    cpu_read_cache: Vec<f32>,
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
        let mut static_gradient_texture_opt = None;
        let mut enable_static_gradient_flag = false;

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture_descriptor = wgpu::TextureDescriptor {
            label: Some("Pheromone Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        let texture_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pheromone Texture A"),
            ..texture_descriptor
        });
        let texture_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pheromone Texture B"),
            ..texture_descriptor
        });

        let initial_data_size =
            (width * height * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress;
        if initial_data_size > 0 {
            const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let bytes_per_pixel = std::mem::size_of::<f32>() as u32;
            let unaligned_bytes_per_row = bytes_per_pixel * width;
            let aligned_bytes_per_row = (unaligned_bytes_per_row + COPY_BYTES_PER_ROW_ALIGNMENT
                - 1)
                & !(COPY_BYTES_PER_ROW_ALIGNMENT - 1);

            let mut padded_initial_data_bytes =
                Vec::with_capacity((aligned_bytes_per_row * height) as usize);
            let single_row_unpadded_zeros_f32 = vec![0.0f32; width as usize];
            let single_row_unpadded_bytes = bytemuck::cast_slice(&single_row_unpadded_zeros_f32);

            for _ in 0..height {
                padded_initial_data_bytes.extend_from_slice(single_row_unpadded_bytes);
                if aligned_bytes_per_row > unaligned_bytes_per_row {
                    padded_initial_data_bytes.resize(
                        padded_initial_data_bytes.len()
                            + (aligned_bytes_per_row - unaligned_bytes_per_row) as usize,
                        0u8,
                    );
                }
            }

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture_a,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded_initial_data_bytes, // Use padded data
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(height),
                },
                texture_size,
            );
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture_b,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded_initial_data_bytes, // Use padded data
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(height),
                },
                texture_size,
            );
        }

        let texture_a_view = texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_b_view = texture_b.create_view(&wgpu::TextureViewDescriptor::default());

        // --- Decay Compute Pipeline Setup ---
        let decay_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decay Shader Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DECAY_SHADER_WGSL)),
        });

        let decay_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Decay Bind Group Layout"),
                entries: &[
                    // Input Texture (Source)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Output Texture (Destination)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Uniforms (DecayFactor)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<DecayUniforms>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let decay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Decay Pipeline Layout"),
                bind_group_layouts: &[&decay_bind_group_layout],
                push_constant_ranges: &[],
            });

        let decay_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Decay Compute Pipeline"),
            layout: Some(&decay_pipeline_layout),
            module: &decay_shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // --- Deposit Compute Pipeline Setup ---
        let deposit_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Deposit Shader Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DEPOSIT_SHADER_WGSL)),
        });

        let deposit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Deposit Bind Group Layout"),
                entries: &[
                    // Pheromone Texture (output, to be written to)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly, // Agents only write
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Agents Buffer (input)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<GpuAgent>() as u64,
                            ), // Min size of one agent
                        },
                        count: None,
                    },
                    // Uniforms (sim_params)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                DepositUniforms,
                            >()
                                as u64),
                        },
                        count: None,
                    },
                ],
            });

        let deposit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Deposit Pipeline Layout"),
                bind_group_layouts: &[&deposit_bind_group_layout],
                push_constant_ranges: &[],
            });

        let deposit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Deposit Compute Pipeline"),
            layout: Some(&deposit_pipeline_layout),
            module: &deposit_shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // --- Diffuse Compute Pipeline Setup ---
        let diffuse_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Diffuse Shader Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(DIFFUSE_SHADER_WGSL)),
        });

        let diffuse_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Diffuse Bind Group Layout"),
                entries: &[
                    // Input Texture (Source)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false }, // Use unfilterable if using textureLoad
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Output Texture (Destination)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Uniforms (DiffuseParams)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                                DiffuseUniforms,
                            >()
                                as u64),
                        },
                        count: None,
                    },
                ],
            });

        let diffuse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Diffuse Pipeline Layout"),
                bind_group_layouts: &[&diffuse_bind_group_layout],
                push_constant_ranges: &[],
            });

        let diffuse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffuse Compute Pipeline"),
            layout: Some(&diffuse_pipeline_layout),
            module: &diffuse_shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

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
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });

                const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
                let bytes_per_pixel_sg = std::mem::size_of::<f32>() as u32; // Assuming f32 for static gradient
                let unaligned_bytes_per_row_sg = bytes_per_pixel_sg * width;
                let aligned_bytes_per_row_sg =
                    (unaligned_bytes_per_row_sg + COPY_BYTES_PER_ROW_ALIGNMENT - 1)
                        & !(COPY_BYTES_PER_ROW_ALIGNMENT - 1);

                let sg_vec_bytes_unpadded = bytemuck::cast_slice(&sg_vec);
                let mut padded_sg_data_bytes =
                    Vec::with_capacity((aligned_bytes_per_row_sg * height) as usize);

                for i in 0..height {
                    let row_start_offset = (i * unaligned_bytes_per_row_sg) as usize;
                    let row_end_offset = row_start_offset + unaligned_bytes_per_row_sg as usize;
                    if row_end_offset <= sg_vec_bytes_unpadded.len() {
                        // Ensure slice is within bounds
                        padded_sg_data_bytes.extend_from_slice(
                            &sg_vec_bytes_unpadded[row_start_offset..row_end_offset],
                        );
                        if aligned_bytes_per_row_sg > unaligned_bytes_per_row_sg {
                            padded_sg_data_bytes.resize(
                                padded_sg_data_bytes.len()
                                    + (aligned_bytes_per_row_sg - unaligned_bytes_per_row_sg)
                                        as usize,
                                0u8,
                            );
                        }
                    } else {
                        // This case should not happen if sg_vec.len() == width * height
                        log::error!(
                            "Error preparing static gradient data: source data too small for row copy."
                        );
                        break;
                    }
                }

                // Check if loop broke early due to error
                if padded_sg_data_bytes.len() == (aligned_bytes_per_row_sg * height) as usize {
                    queue.write_texture(
                        wgpu::ImageCopyTexture {
                            texture: &static_texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        &padded_sg_data_bytes, // Use padded data
                        wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(aligned_bytes_per_row_sg),
                            rows_per_image: Some(height),
                        },
                        texture_size,
                    );
                    static_gradient_texture_opt = Some(static_texture);
                    enable_static_gradient_flag = true;
                } else {
                    log::error!(
                        "Static gradient texture not written due to data preparation error."
                    );
                }
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

        Self {
            decay_factor,
            enable_static_gradient: enable_static_gradient_flag,
            enable_dynamic_gradient,
            texture_a,
            texture_b,
            texture_a_view,
            texture_b_view,
            current_texture_is_a: true,
            static_gradient_texture: static_gradient_texture_opt,
            height,
            width,
            device,
            queue,
            decay_pipeline,
            decay_bind_group_layout,
            deposit_pipeline,
            deposit_bind_group_layout,
            diffuse_pipeline,
            diffuse_bind_group_layout,
            cpu_read_cache: vec![0.0f32; (width * height) as usize],
        }
    }

    /// Updates the CPU-side cache (`cpu_read_cache`) with the current GPU pheromone data.
    /// This should be called after all pheromone GPU operations for a frame are complete.
    pub fn update_cpu_read_cache(&mut self) {
        if !self.enable_dynamic_gradient && !self.enable_static_gradient {
            // If no gradients are enabled, cache can remain zero or empty, effectively.
            // Or, ensure it's explicitly cleared if necessary.
            // For now, if it was initialized to zeros and dynamic is off, it means no dynamic pheromones.
            // If static is also off, then readings should be 0. If static is on, get_reading handles it.
            // This primarily ensures we don't do a pointless GPU readback if dynamic is off.
            // However, if static is on and dynamic is off, `read_current_texture_to_vec` would read the (empty) dynamic texture.
            // The current `read_current_texture_to_vec` reads the dynamic textures (texture_a/b).
            // If dynamic_gradient is off, these textures aren't updated by deposit/diffuse/decay.
            // So, if only static is enabled, this cache will reflect empty dynamic values, which is correct.
            return; // Avoid GPU readback if dynamic pheromones are not being updated.
        }
        self.cpu_read_cache = self.read_current_texture_to_vec();
    }

    pub fn get_reading(&self, at_location: Point2) -> Option<i32> {
        let (x_f, y_f) = (at_location.x.round(), at_location.y.round());
        let is_within_bounds =
            x_f >= 0.0 && x_f < self.width as f32 && y_f >= 0.0 && y_f < self.height as f32;

        if is_within_bounds {
            let x_u32 = x_f as u32;
            let y_u32 = y_f as u32;

            // Calculate 1D index for the row-major cpu_read_cache
            let index = (y_u32 * self.width + x_u32) as usize;
            let mut dynamic_pheromone_value = 0.0f32;

            if self.enable_dynamic_gradient {
                if index < self.cpu_read_cache.len() {
                    dynamic_pheromone_value = self.cpu_read_cache[index];
                } else {
                    // This case should ideally not happen if coordinates are within bounds
                    // and cache is sized correctly.
                    log::warn!(
                        "Index out of bounds for cpu_read_cache. Index: {}, Cache size: {}. Coords: ({}, {})",
                        index,
                        self.cpu_read_cache.len(),
                        x_u32,
                        y_u32
                    );
                }
            }

            match (self.enable_dynamic_gradient, self.enable_static_gradient) {
                (true, true) => {
                    let pheromone_value = dynamic_pheromone_value;
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
                    let static_value = static_value_f32_opt.unwrap_or(0.0);
                    Some(((static_value + pheromone_value) * 255.0).round() as i32)
                }
                (true, false) => Some((dynamic_pheromone_value * 255.0).round() as i32),
                (false, true) => {
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
        if !self.enable_dynamic_gradient || agents.is_empty() {
            // If not dynamic or no agents, this operation effectively does nothing to agent depositions.
            // However, to maintain the ping-pong texture flow for subsequent diffuse/decay,
            // we still need to ensure the "next" texture has the data from the "current" one
            // and then swap them.
            let (current_texture, _current_view) = self.get_current_active_texture();
            let (next_texture, _next_view) = self.get_next_texture_and_view();

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Deposit No-Op Copy Encoder"),
                });

            encoder.copy_texture_to_texture(
                current_texture.as_image_copy(),
                next_texture.as_image_copy(),
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));
            self.current_texture_is_a = !self.current_texture_is_a; // Swap current and next
            return;
        }

        let (current_texture, _current_view) = self.get_current_active_texture();
        let (_next_texture, next_view) = self.get_next_texture_and_view(); // Shader writes to next_view

        // Prepare agent data for GPU
        let gpu_agents: Vec<GpuAgent> = agents
            .iter()
            .map(|agent| GpuAgent {
                position: [agent.location().x, agent.location().y],
                deposition_amount: agent.deposition_amount(),
                _padding: 0.0, // Padding for 16-byte alignment
            })
            .collect();

        let agent_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Agent Buffer (Deposit)"),
                contents: bytemuck::cast_slice(&gpu_agents),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let uniforms = DepositUniforms {
            texture_width: self.width,
            texture_height: self.height,
            num_agents: agents.len() as u32,
            _padding: 0, // Padding for UBO
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Deposit Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let deposit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Deposit Bind Group"),
            layout: &self.deposit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(next_view), // Shader writes to the 'next' texture view
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Deposit Command Encoder"),
            });

        // Step 1: Copy current texture to next texture (so deposit shader starts with previous state)
        encoder.copy_texture_to_texture(
            current_texture.as_image_copy(),
            self.get_next_texture_and_view().0.as_image_copy(), // ensure it's the texture, not view
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        // Step 2: Run deposit compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Deposit Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.deposit_pipeline);
            compute_pass.set_bind_group(0, &deposit_bind_group, &[]);

            // Dispatch based on number of agents. Workgroup size X is 64 in shader.
            let dispatch_x = (agents.len() as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(dispatch_x, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Swap textures: the 'next' texture (which was written to) becomes the 'current'
        self.current_texture_is_a = !self.current_texture_is_a;
    }

    pub fn diffuse(&mut self) {
        if !self.enable_dynamic_gradient {
            // If not dynamic, diffuse does nothing. To maintain texture flow, copy current to next & swap.
            let (current_texture, _current_view) = self.get_current_active_texture();
            let (next_texture, _next_view) = self.get_next_texture_and_view();

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Diffuse No-Op Copy Encoder"),
                });
            encoder.copy_texture_to_texture(
                current_texture.as_image_copy(),
                next_texture.as_image_copy(),
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(std::iter::once(encoder.finish()));
            self.current_texture_is_a = !self.current_texture_is_a; // Swap current and next
            return;
        }
        // log::warn!("Pheromones::diffuse() called but not yet fully implemented for textures beyond copy.");

        let (source_view, dest_view) = if self.current_texture_is_a {
            (&self.texture_a_view, &self.texture_b_view)
        } else {
            (&self.texture_b_view, &self.texture_a_view)
        };

        let uniforms = DiffuseUniforms {
            texture_width: self.width,
            texture_height: self.height,
            _padding1: 0,
            _padding2: 0,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Diffuse Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let diffuse_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diffuse Bind Group"),
            layout: &self.diffuse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dest_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Diffuse Command Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Diffuse Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.diffuse_pipeline);
            compute_pass.set_bind_group(0, &diffuse_bind_group, &[]);

            // Dispatch based on texture dimensions and workgroup size (8x8)
            let dispatch_width = (self.width + 7) / 8;
            let dispatch_height = (self.height + 7) / 8;
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Swap textures for next operation
        self.current_texture_is_a = !self.current_texture_is_a;
    }

    pub fn decay(&mut self) {
        if !self.enable_dynamic_gradient {
            return;
        }
        // log::warn!("Pheromones::decay() called but not yet implemented for textures.");
        // self.current_texture_is_a = !self.current_texture_is_a; // Old placeholder

        let (source_view, dest_view) = if self.current_texture_is_a {
            (&self.texture_a_view, &self.texture_b_view)
        } else {
            (&self.texture_b_view, &self.texture_a_view)
        };

        let uniforms = DecayUniforms {
            decay_factor: self.decay_factor,
        };
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Decay Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, // COPY_DST if we update it often
            });

        let decay_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Decay Bind Group"),
            layout: &self.decay_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dest_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Decay Command Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Decay Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.decay_pipeline);
            compute_pass.set_bind_group(0, &decay_bind_group, &[]);

            // Dispatch based on texture dimensions and workgroup size (8x8)
            let dispatch_width = (self.width + 7) / 8; // Equivalent to ceil(width / 8.0)
            let dispatch_height = (self.height + 7) / 8; // Equivalent to ceil(height / 8.0)
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        // Swap textures for next frame
        self.current_texture_is_a = !self.current_texture_is_a;
    }

    pub fn set_decay_factor(&mut self, decay_factor: f32) {
        self.decay_factor = decay_factor;
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        log::warn!(
            "Pheromones::iter() called but not yet implemented for textures. Returning empty iterator."
        );
        std::iter::empty()
    }

    pub fn enable_dynamic_gradient(&mut self) {
        self.enable_dynamic_gradient = true;
    }

    pub fn disable_dynamic_gradient(&mut self) {
        self.enable_dynamic_gradient = false;
    }

    pub fn get_current_grid(&self) -> &DMatrix<f32> {
        // self.grid.a() // Old DMatrix access
        // This is a major breaking change. World::draw expects &DMatrix<f32>.
        // We'll need to return something else, or have World::draw adapt.
        panic!(
            "Pheromones::get_current_grid() is not compatible with wgpu::Texture. It needs to be redesigned to provide texture data for drawing."
        );
        // The function signature itself (returning &DMatrix) is now incorrect.
        // This will need to change to return perhaps a Vec<f32> or Vec<u8> after reading from texture,
        // or World::draw will need to take &Pheromones and handle texture reading itself.
    }

    fn get_current_active_texture(&self) -> (&wgpu::Texture, &wgpu::TextureView) {
        if self.current_texture_is_a {
            (&self.texture_a, &self.texture_a_view)
        } else {
            (&self.texture_b, &self.texture_b_view)
        }
    }

    #[allow(dead_code)] // To be used by compute shaders later
    fn get_next_texture_and_view(&self) -> (&wgpu::Texture, &wgpu::TextureView) {
        if self.current_texture_is_a {
            // Current is A, next is B
            (&self.texture_b, &self.texture_b_view)
        } else {
            // Current is B, next is A
            (&self.texture_a, &self.texture_a_view)
        }
    }

    /// Reads the currently active pheromone texture data into a `Vec<f32>`.
    /// This is a synchronous operation that will block until the GPU readback is complete.
    /// The returned Vec contains the texture data in row-major order.
    pub fn read_current_texture_to_vec(&self) -> Vec<f32> {
        let (current_texture, _view) = self.get_current_active_texture();

        let texture_format = current_texture.format();
        let bytes_per_pixel = texture_format
            .block_copy_size(Some(wgpu::TextureAspect::All))
            .unwrap_or_else(|| {
                panic!(
                    "Could not determine block_copy_size for texture format {:?}",
                    texture_format
                )
            });

        if bytes_per_pixel == 0 {
            panic!(
                "Texture format {:?} has a block_copy_size of 0.",
                texture_format
            );
        }

        const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unaligned_bytes_per_row = bytes_per_pixel * self.width;
        let aligned_bytes_per_row = (unaligned_bytes_per_row + COPY_BYTES_PER_ROW_ALIGNMENT - 1)
            & !(COPY_BYTES_PER_ROW_ALIGNMENT - 1);

        // The buffer needs to be large enough for the COPY, considering aligned rows.
        let buffer_size = aligned_bytes_per_row as u64 * self.height as u64;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pheromone Readback Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Pheromone Readback Encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: current_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(self.height), // This should be self.height for full texture
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).expect("Failed to send map_async result");
        });
        self.device.poll(wgpu::Maintain::Wait); // Block until operation is complete

        rx.recv()
            .expect("Failed to receive map_async result from channel")
            .expect("Failed to map staging buffer");

        let data = buffer_slice.get_mapped_range();

        // Create a new Vec to hold the tightly packed data
        let mut result_vec: Vec<f32> = Vec::with_capacity((self.width * self.height) as usize);
        for row in 0..self.height {
            let offset = (row * aligned_bytes_per_row) as usize;
            let row_data = &data[offset..(offset + (self.width * bytes_per_pixel) as usize)];
            result_vec.extend_from_slice(bytemuck::cast_slice(row_data));
        }

        drop(data);
        staging_buffer.unmap();

        result_vec
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
