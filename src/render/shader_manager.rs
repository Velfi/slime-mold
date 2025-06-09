use std::borrow::Cow;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

pub struct ShaderManager {
    pub compute_shader: ShaderModule,
    pub display_shader: ShaderModule,
    pub gradient_shader: ShaderModule,
    pub quad_shader: ShaderModule,
}

impl ShaderManager {
    pub fn new(device: &Device) -> Self {
        Self {
            compute_shader: Self::create_shader(
                device,
                "Compute Shader",
                include_str!("../shaders/compute.wgsl"),
            ),
            display_shader: Self::create_shader(
                device,
                "Display Compute Shader",
                include_str!("../shaders/display.wgsl"),
            ),
            gradient_shader: Self::create_shader(
                device,
                "Gradient Compute Shader",
                include_str!("../shaders/gradient.wgsl"),
            ),
            quad_shader: Self::create_shader(
                device,
                "Quad Shader",
                include_str!("../shaders/quad.wgsl"),
            ),
        }
    }

    fn create_shader(device: &Device, label: &str, source: &str) -> ShaderModule {
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(Cow::Borrowed(source)),
        })
    }
}
