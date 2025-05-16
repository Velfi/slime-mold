// display.wgsl: Use uniform for width/height

@group(0) @binding(0)
var<storage, read> trail_map: array<f32>;
@group(0) @binding(1)
var display_tex: texture_storage_2d<rgba8unorm, write>;

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
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};
@group(0) @binding(2)
var<uniform> sim_size: SimSizeUniform;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    if (x >= i32(sim_size.width) || y >= i32(sim_size.height)) {
        return;
    }
    let idx = y * i32(sim_size.width) + x;
    let intensity = clamp(trail_map[idx], 0.0, 1.0);
    let color = vec4<f32>(intensity, intensity, intensity, 1.0);
    
    // Get texture dimensions
    let texture_width = textureDimensions(display_tex).x;
    let texture_height = textureDimensions(display_tex).y;
    
    // Calculate scaling factors
    let scale_x = f32(texture_width) / f32(sim_size.width);
    let scale_y = f32(texture_height) / f32(sim_size.height);
    
    // Calculate scaled coordinates
    let scaled_x = i32(f32(x) * scale_x);
    let scaled_y = i32(f32(y) * scale_y);
    
    // Write to all pixels in the scaled region
    let next_x = i32(f32(x + 1) * scale_x);
    let next_y = i32(f32(y + 1) * scale_y);
    
    for (var ty = scaled_y; ty < next_y; ty = ty + 1) {
        for (var tx = scaled_x; tx < next_x; tx = tx + 1) {
            if (tx >= 0 && tx < i32(texture_width) && ty >= 0 && ty < i32(texture_height)) {
                textureStore(display_tex, vec2<i32>(tx, ty), color);
            }
        }
    }
} 