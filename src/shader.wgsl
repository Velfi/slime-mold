struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Vertex positions and UVs for a full-screen quad (two triangles)
    // T1: (-1,1) (0,0), (-1,-1) (0,1), (1,1) (1,0)
    // T2: (-1,-1) (0,1), (1,-1) (1,1), (1,1) (1,0)

    if (in_vertex_index == 0u) { // Top-left of T1
        out.clip_position = vec4<f32>(-1.0, 1.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(0.0, 0.0);
    } else if (in_vertex_index == 1u) { // Bottom-left of T1 & T2
        out.clip_position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(0.0, 1.0);
    } else if (in_vertex_index == 2u) { // Top-right of T1 & T2
        out.clip_position = vec4<f32>(1.0, 1.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(1.0, 0.0);
    } else if (in_vertex_index == 3u) { // Bottom-left of T2 (same as index 1)
        out.clip_position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(0.0, 1.0);
    } else if (in_vertex_index == 4u) { // Bottom-right of T2
        out.clip_position = vec4<f32>(1.0, -1.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(1.0, 1.0);
    } else { // (in_vertex_index == 5u) Top-right of T2 (same as index 2)
        out.clip_position = vec4<f32>(1.0, 1.0, 0.0, 1.0);
        out.tex_coords = vec2<f32>(1.0, 0.0);
    }

    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
} 