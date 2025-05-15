struct GpuAgent {
    position: vec2<f32>,
    deposition_amount: f32,
    // No padding needed here if tightly packed in Rust and just an array of these.
    // If Rust side uses padding for 16-byte alignment per agent, match it here.
};

@group(0) @binding(0) var pheromone_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<storage, read> agents_buffer: array<GpuAgent>;

struct Uniforms {
    texture_width: u32,
    texture_height: u32,
    num_agents: u32,
    _padding: u32, // Ensure 16-byte alignment for the UBO struct
};
@group(0) @binding(2) var<uniform> sim_params: Uniforms;

@compute @workgroup_size(64, 1, 1) // Process 64 agents per workgroup
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let agent_index = global_id.x;

    if (agent_index >= sim_params.num_agents) {
        return;
    }

    let agent = agents_buffer[agent_index];
    let agent_pos = agent.position;
    let deposition_amount = agent.deposition_amount;

    // Wrap coordinates (same logic as CPU version previously)
    // Ensure f32 for modulo operations involving floats
    let x_wrapped = (agent_pos.x % f32(sim_params.texture_width) + f32(sim_params.texture_width)) % f32(sim_params.texture_width);
    let y_wrapped = (agent_pos.y % f32(sim_params.texture_height) + f32(sim_params.texture_height)) % f32(sim_params.texture_height);

    let texel_coord = vec2<i32>(i32(x_wrapped), i32(y_wrapped));

    // Boundary check for texel_coord before textureStore (optional if wrapping is perfect and within u32 range)
    // However, agent_pos could be negative, ensure x_wrapped/y_wrapped are positive before casting to i32 for texel_coord.
    // The modulo logic should handle this if width/height are positive.
    // A robust check:
    if (texel_coord.x >= 0 && texel_coord.x < i32(sim_params.texture_width) && 
        texel_coord.y >= 0 && texel_coord.y < i32(sim_params.texture_height)) {
        textureStore(pheromone_texture, texel_coord, vec4<f32>(deposition_amount, 0.0, 0.0, 0.0));
    }
} 