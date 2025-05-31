use std::borrow::Cow;
use wgpu::{Device, ShaderModule, ShaderModuleDescriptor, ShaderSource};

pub struct ShaderManager {
    pub compute_shader: ShaderModule,
    pub display_shader: ShaderModule,
    pub quad_shader: ShaderModule,
    pub text_shader: ShaderModule,
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
            quad_shader: Self::create_shader(device, "Quad Shader", include_str!("../shaders/quad.wgsl")),
            text_shader: Self::create_shader(device, "Text Shader", include_str!("../shaders/text.wgsl")),
        }
    }

    fn create_shader(device: &Device, label: &str, source: &str) -> ShaderModule {
        device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(Cow::Borrowed(source)),
        })
    }
}
