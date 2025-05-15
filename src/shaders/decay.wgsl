@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var dest_texture: texture_storage_2d<r32float, write>;

struct DecayParams {
    decay_factor: f32,
    // We can add width/height here if needed, but for simple decay, it's not.
};
@group(0) @binding(2) var<uniform> decay_params: DecayParams;

// const WORKGROUP_SIZE_X: u32 = 8u; // Cannot use const in workgroup_size
// const WORKGROUP_SIZE_Y: u32 = 8u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tex_dims = textureDimensions(source_texture);

    // Alternative boundary check logic
    let out_of_bounds_x: bool = u32(global_id.x) >= u32(tex_dims.x);
    let out_of_bounds_y: bool = u32(global_id.y) >= u32(tex_dims.y);

    if (out_of_bounds_x || out_of_bounds_y) {
        return;
    }

    let texel_coord = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let current_value = textureLoad(source_texture, texel_coord, 0); // 0 for mip level

    let decayed_value = current_value.x * (1.0 - decay_params.decay_factor);

    textureStore(dest_texture, texel_coord, vec4<f32>(decayed_value, 0.0, 0.0, 0.0));
} 