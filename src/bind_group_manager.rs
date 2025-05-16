use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, Buffer, Device, TextureView,
};

pub struct BindGroupManager {
    pub compute_bind_group: BindGroup,
    pub display_bind_group: BindGroup,
    pub render_bind_group: BindGroup,
}

impl BindGroupManager {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        compute_bind_group_layout: &BindGroupLayout,
        display_bind_group_layout: &BindGroupLayout,
        render_bind_group_layout: &BindGroupLayout,
        agent_buffer: &Buffer,
        trail_map_buffer: &Buffer,
        sim_size_buffer: &Buffer,
        display_view: &TextureView,
        display_sampler: &wgpu::Sampler,
        lut_buffer: &Buffer,
    ) -> Self {
        Self {
            compute_bind_group: Self::create_compute_bind_group(
                device,
                compute_bind_group_layout,
                agent_buffer,
                trail_map_buffer,
                sim_size_buffer,
            ),
            display_bind_group: Self::create_display_bind_group(
                device,
                display_bind_group_layout,
                trail_map_buffer,
                display_view,
                sim_size_buffer,
                lut_buffer,
            ),
            render_bind_group: Self::create_render_bind_group(
                device,
                render_bind_group_layout,
                display_view,
                display_sampler,
            ),
        }
    }

    pub fn update_compute_bind_group(
        &mut self,
        device: &Device,
        compute_bind_group_layout: &BindGroupLayout,
        agent_buffer: &Buffer,
        trail_map_buffer: &Buffer,
        sim_size_buffer: &Buffer,
    ) {
        self.compute_bind_group = Self::create_compute_bind_group(
            device,
            compute_bind_group_layout,
            agent_buffer,
            trail_map_buffer,
            sim_size_buffer,
        );
    }

    pub fn update_display_bind_group(
        &mut self,
        device: &Device,
        display_bind_group_layout: &BindGroupLayout,
        trail_map_buffer: &Buffer,
        display_view: &TextureView,
        sim_size_buffer: &Buffer,
        lut_buffer: &Buffer,
    ) {
        self.display_bind_group = Self::create_display_bind_group(
            device,
            display_bind_group_layout,
            trail_map_buffer,
            display_view,
            sim_size_buffer,
            lut_buffer,
        );
    }

    pub fn update_render_bind_group(
        &mut self,
        device: &Device,
        render_bind_group_layout: &BindGroupLayout,
        display_view: &TextureView,
        display_sampler: &wgpu::Sampler,
    ) {
        self.render_bind_group = Self::create_render_bind_group(
            device,
            render_bind_group_layout,
            display_view,
            display_sampler,
        );
    }

    fn create_compute_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        agent_buffer: &Buffer,
        trail_map_buffer: &Buffer,
        sim_size_buffer: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: agent_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: trail_map_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: sim_size_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_display_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        trail_map_buffer: &Buffer,
        display_view: &TextureView,
        sim_size_buffer: &Buffer,
        lut_buffer: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("Display Compute Bind Group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: trail_map_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(display_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: sim_size_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: lut_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn create_render_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        display_view: &TextureView,
        display_sampler: &wgpu::Sampler,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(display_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(display_sampler),
                },
            ],
        })
    }
}
