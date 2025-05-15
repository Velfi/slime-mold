@group(0) @binding(0) var<storage, read> pheromone_values: array<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1) // Workgroup size can be tuned
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_dims = textureDimensions(output_texture);
    let td_x: u32 = u32(texture_dims.x);
    let td_y: u32 = u32(texture_dims.y);

    // Boundary check
    if (global_id.x >= td_x || global_id.y >= td_y) {
        return;
    }

    let flat_index = global_id.y * td_x + global_id.x;

    let pheromone_value = pheromone_values[flat_index];
    
    let val_scaled = pheromone_value * 255.0;
    let val_clamped = clamp(val_scaled, 0.0, 255.0);
    
    let normalized_val = val_clamped / 255.0;

    textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(normalized_val, normalized_val, normalized_val, 1.0));
} 