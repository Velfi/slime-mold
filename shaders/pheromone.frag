#version 450

layout(location = 0) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform Uniforms {
    float pheromone_decay_factor;
    float pheromone_diffusion_rate;
    float delta_time;
};

layout(set = 0, binding = 1) uniform sampler2D t_pheromone;
layout(set = 0, binding = 2) uniform sampler2D t_deposition;

void main() {
    // Sample current pheromone value
    vec4 current = texture(t_pheromone, v_tex_coords);
    
    // Sample deposition from agents
    vec4 deposition = texture(t_deposition, v_tex_coords);
    
    // Apply decay
    float decay = 1.0 - (pheromone_decay_factor * delta_time);
    current.r *= decay;
    
    // Optimized diffusion using bilinear filtering
    vec2 texel_size = 1.0 / textureSize(t_pheromone, 0);
    vec2 offset = texel_size * 0.5; // Half texel offset for bilinear filtering
    
    // Sample 4 corners instead of 8 neighbors
    vec4 corners = vec4(0.0);
    corners += texture(t_pheromone, v_tex_coords + vec2(-1, -1) * offset);
    corners += texture(t_pheromone, v_tex_coords + vec2(-1,  1) * offset);
    corners += texture(t_pheromone, v_tex_coords + vec2( 1, -1) * offset);
    corners += texture(t_pheromone, v_tex_coords + vec2( 1,  1) * offset);
    
    // Mix current value with diffused value and add deposition in one step
    float diffusion = pheromone_diffusion_rate * delta_time;
    current.r = mix(current.r, corners.r * 0.25, diffusion) + deposition.r;
    
    // Output final pheromone value
    f_color = current;
} 