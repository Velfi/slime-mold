#version 450

layout(location = 0) in vec2 v_position;
layout(location = 1) in vec2 v_velocity;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform Uniforms {
    float pheromone_deposition_amount;
    float delta_time;
};

void main() {
    // Calculate pheromone deposition based on agent's position and velocity
    float deposition = pheromone_deposition_amount * delta_time;
    
    // Output the pheromone deposition value
    f_color = vec4(deposition, 0.0, 0.0, 1.0);
} 