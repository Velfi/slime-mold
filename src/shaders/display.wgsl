// Display shader for converting trail map to displayable texture
// Uses LUT for color mapping

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

@group(0) @binding(0)
var<storage, read> trail_map: array<f32>;

@group(0) @binding(1)
var display_tex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> sim_size: SimSizeUniform;

@group(0) @binding(3)
var<storage, read> lut_data: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    if (x >= i32(sim_size.width) || y >= i32(sim_size.height)) {
        return;
    }

    let idx = y * i32(sim_size.width) + x;
    let value = trail_map[idx];

    // Map value to LUT index (0-255)
    let lut_idx = u32(clamp(value * 255.0, 0.0, 255.0));
    
    // Get RGB values from LUT
    let r = f32(lut_data[lut_idx]) / 255.0;
    let g = f32(lut_data[lut_idx + 256]) / 255.0;
    let b = f32(lut_data[lut_idx + 512]) / 255.0;

    // Write to display texture
    textureStore(display_tex, vec2<i32>(x, y), vec4<f32>(r, g, b, 1.0));
} 