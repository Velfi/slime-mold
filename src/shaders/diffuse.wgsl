@group(0) @binding(0) var source_texture: texture_2d<f32>; // Input
@group(0) @binding(1) var dest_texture: texture_storage_2d<r32float, write>; // Output

struct DiffuseParams {
    texture_width: u32,
    texture_height: u32,
    // x_radius and y_radius are effectively 1 for a 3x3 box blur
    // No explicit radius uniform needed if fixed 3x3, but could be added for flexibility
    _padding1: u32, // Padding for UBO alignment if other fields were added
    _padding2: u32,
};
@group(0) @binding(2) var<uniform> params: DiffuseParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.texture_width || y >= params.texture_height) {
        return;
    }

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var count = 0.0;

    // 3x3 Box Blur (radius 1)
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            // Calculate wrapped coordinates for sampling
            let sample_x = i32(x) + i;
            let sample_y = i32(y) + j;

            // Simple clamp to edge for this example. For slime mold, wrapping might be better.
            let clamped_x = clamp(sample_x, 0, i32(params.texture_width) - 1);
            let clamped_y = clamp(sample_y, 0, i32(params.texture_height) - 1);
            
            // If using wrapping instead of clamp:
            // let wrapped_x = (sample_x % i32(params.texture_width) + i32(params.texture_width)) % i32(params.texture_width);
            // let wrapped_y = (sample_y % i32(params.texture_height) + i32(params.texture_height)) % i32(params.texture_height);
            // sum = sum + textureLoad(source_texture, vec2<i32>(wrapped_x, wrapped_y), 0);

            sum = sum + textureLoad(source_texture, vec2<i32>(clamped_x, clamped_y), 0);
            count = count + 1.0;
        }
    }

    let blurred_value = sum / count;
    textureStore(dest_texture, vec2<i32>(i32(x), i32(y)), blurred_value);
} 