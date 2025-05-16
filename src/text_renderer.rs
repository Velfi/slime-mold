use fontdue::Font;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, Buffer, Device, Queue, Sampler, Texture};
use winit::dpi::PhysicalSize;

pub struct TextRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    text_size: f32,
    window_height: u32,
    text_texture: Option<Texture>,
    text_bind_group: Option<BindGroup>,
    text_bind_group_layout: BindGroupLayout,
    text_sampler: Sampler,
    uniform_buffer: Arc<Buffer>,
    lut_buffer: Arc<Buffer>,
    visible: bool,
}

impl TextRenderer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        window_height: u32,
        uniform_buffer: Arc<Buffer>,
        lut_buffer: Arc<Buffer>,
    ) -> Self {
        // Create bind group layout for text rendering
        let text_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Text Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create sampler for text texture
        let text_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Text Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            device,
            queue,
            text_size: window_height as f32 / 40.0,
            window_height,
            text_texture: None,
            text_bind_group: None,
            text_bind_group_layout,
            text_sampler,
            uniform_buffer,
            lut_buffer,
            visible: true,
        }
    }

    pub fn update_window_size(&mut self, window_height: u32) {
        self.window_height = window_height;
        self.text_size = window_height as f32 / 40.0;
    }

    pub fn render_text(&mut self, text: &str, font: &Font, window_size: PhysicalSize<u32>) {
        // Update text size if window height changed
        if window_size.height != self.window_height {
            self.update_window_size(window_size.height);
        }

        // Calculate text layout
        let line_height = self.text_size * 1.5;
        let mut lines: Vec<&str> = text.lines().collect();
        lines.reverse();

        // Calculate total height of text block
        let total_height = lines.len() as f32 * line_height;

        // Add padding around text
        let padding = self.text_size * 0.5; // Padding is half the text size

        // Calculate the maximum width of all lines for the background box
        let mut max_width = 0.0;
        let space_advance = font.metrics(' ', self.text_size).advance_width;
        let tab_spaces = 4;
        let tab_width = space_advance * tab_spaces as f32;
        for line in lines.iter() {
            let mut line_width = 0.0;
            for ch in line.chars() {
                if ch == '\t' {
                    // Advance to next tab stop
                    let next_tab = ((line_width / tab_width).floor() + 1.0) * tab_width;
                    line_width = next_tab;
                } else {
                    let metrics = font.metrics(ch, self.text_size);
                    line_width += metrics.advance_width;
                }
            }
            max_width = f32::max(max_width, line_width);
        }

        // Calculate starting positions with padding
        let start_x = padding;
        let start_y = (window_size.height as f32 - total_height) / 2.0;

        // Create a texture to hold the text
        let texture_width = window_size.width;
        let texture_height = window_size.height;

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Text Texture"),
            size: wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Create a buffer to hold the text bitmap (4 bytes per pixel for RGBA)
        let mut text_bitmap = vec![0u8; (texture_width * texture_height * 4) as usize];

        // Draw the background box with padding
        let box_left = (start_x - padding) as usize;
        let box_top = (start_y - padding) as usize;
        let box_width = (max_width + padding * 2.0) as usize;
        let box_height = (total_height + padding * 2.0) as usize;

        // Fill the background box area with a solid black background
        for y in box_top..box_top + box_height {
            for x in box_left..box_left + box_width {
                if y < texture_height as usize && x < texture_width as usize {
                    let idx = (y * texture_width as usize + x) * 4;
                    if idx + 3 < text_bitmap.len() {
                        // RGBA: (0,0,0,255) for solid black
                        text_bitmap[idx] = 0; // R = 0
                        text_bitmap[idx + 1] = 0; // G = 0
                        text_bitmap[idx + 2] = 0; // B = 0
                        text_bitmap[idx + 3] = 255; // A = 255 (fully opaque)
                    }
                }
            }
        }

        // Render each line of text
        for (i, line) in lines.iter().enumerate() {
            let baseline_y = start_y + i as f32 * line_height + self.text_size;
            let mut x_position = start_x;

            for ch in line.chars() {
                if ch == '\t' {
                    // Advance to next tab stop
                    let next_tab = ((x_position / tab_width).floor() + 1.0) * tab_width;
                    x_position = next_tab;
                    continue;
                }
                let (raster_metrics, bitmap) = font.rasterize(ch, self.text_size);
                let font_metrics = font.metrics(ch, self.text_size);

                let glyph_y = baseline_y + font_metrics.bounds.ymin;

                // Copy bitmap to our texture data
                for y in 0..raster_metrics.height {
                    for x in 0..raster_metrics.width {
                        let src_y = raster_metrics.height - 1 - y;
                        let src_idx = src_y * raster_metrics.width + x;

                        let dst_x = x_position as usize + x;
                        let dst_y = glyph_y as usize + y;

                        if dst_y < texture_height as usize && dst_x < texture_width as usize {
                            let dst_idx = (dst_y * texture_width as usize + dst_x) * 4;
                            if dst_idx + 3 < text_bitmap.len()
                                && src_idx < bitmap.len()
                                && bitmap[src_idx] > 0
                            {
                                // White text (255,255,255) with alpha from the font rasterizer
                                text_bitmap[dst_idx] = 255; // R = 255
                                text_bitmap[dst_idx + 1] = 255; // G = 255
                                text_bitmap[dst_idx + 2] = 255; // B = 255
                                text_bitmap[dst_idx + 3] = bitmap[src_idx]; // A from font
                            }
                        }
                    }
                }

                x_position += font_metrics.advance_width;
            }
        }

        // Upload texture data
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &text_bitmap,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(texture_width * 4), // 4 bytes per pixel for RGBA
                rows_per_image: Some(texture_height),
            },
            wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );

        // Create bind group for text texture
        let text_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Text Bind Group"),
            layout: &self.text_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.text_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.lut_buffer.as_entire_binding(),
                },
            ],
        });

        self.text_texture = Some(texture);
        self.text_bind_group = Some(text_bind_group);
    }

    pub fn toggle_visibility(&mut self) {
        self.visible = !self.visible;
    }

    pub fn get_bind_group(&self) -> Option<&BindGroup> {
        if self.visible {
            self.text_bind_group.as_ref()
        } else {
            None
        }
    }
}
