struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    var tex = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
    output.tex_coords = tex[vertex_index];
    return output;
}

@group(0) @binding(0)
var<uniform> sim_size: SimSizeUniform;

@group(0) @binding(1)
var text_tex: texture_2d<f32>;

@group(0) @binding(2)
var text_sampler: sampler;

@group(0) @binding(3)
var<storage, read> lut_data: array<u32>;

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
    _pad1: u32,
};

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let text_color = textureSample(text_tex, text_sampler, input.tex_coords);
    
    // If the text pixel is transparent, return transparent
    if (text_color.a < 0.1) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Otherwise return the text color
    return text_color;
} 