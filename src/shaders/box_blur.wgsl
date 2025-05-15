@group(0) @binding(0) var<storage, read> input_image: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<f32>;

struct Uniforms {
    width: u32,
    height: u32,
    x_radius: u32,
    y_radius: u32,
};
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_out = global_id.x;
    let y_out = global_id.y;

    if (x_out >= uniforms.width || y_out >= uniforms.height) {
        return;
    }

    var sum: f32 = 0.0;
    let kernel_width: u32 = 2u * uniforms.x_radius + 1u;
    let kernel_height: u32 = 2u * uniforms.y_radius + 1u;
    let kernel_size: f32 = f32(kernel_width * kernel_height);

    for (var ky: u32 = 0u; ky < kernel_height; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < kernel_width; kx = kx + 1u) {
            let x_in_intermediate: i32 = i32(x_out) + i32(kx) - i32(uniforms.x_radius);
            let y_in_intermediate: i32 = i32(y_out) + i32(ky) - i32(uniforms.y_radius);

            // Wrap coordinates
            let x_in = (x_in_intermediate % i32(uniforms.width) + i32(uniforms.width)) % i32(uniforms.width);
            let y_in = (y_in_intermediate % i32(uniforms.height) + i32(uniforms.height)) % i32(uniforms.height);

            let index_in: u32 = u32(y_in) * uniforms.width + u32(x_in);
            sum = sum + input_image[index_in];
        }
    }
    let avg: f32 = sum / kernel_size;
    let index_out: u32 = y_out * uniforms.width + x_out;
    output_image[index_out] = avg;
} 