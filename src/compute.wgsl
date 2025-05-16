// Compute shader for Physarum simulation
// Each agent is represented by a vec4<f32>: x, y, angle, speed

const TAU: f32 = 6.28318530718; // 2π

@group(0) @binding(0)
var<storage, read_write> agents: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> trail_map: array<f32>;

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
@group(0) @binding(2)
var<uniform> sim_size: SimSizeUniform;

// Parameters for the simulation (now mostly from uniform)
const TIME_STEP: f32 = 0.016; // Affects how far agents move per frame based on their speed

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Calculate agent index from 2D workgroup layout
    let agent_index = i32(global_id.x + global_id.y * 65535u);
    if (agent_index >= i32(arrayLength(&agents))) {
        return;
    }

    var agent = agents[agent_index];
    var x = agent.x;
    var y = agent.y;
    var angle = agent.z;
    var speed = agent.w;

    // Ensure speed stays within bounds
    speed = clamp(speed, sim_size.agent_speed_min, sim_size.agent_speed_max);

    // Calculate sensor positions
    let sensor_angle_left = angle - sim_size.agent_sensor_angle;
    let sensor_angle_right = angle + sim_size.agent_sensor_angle;
    let sensor_angle_center = angle;

    let sensor_pos_left = vec2<i32>(
        i32(x + sim_size.agent_sensor_distance * cos(sensor_angle_left)),
        i32(y + sim_size.agent_sensor_distance * sin(sensor_angle_left))
    );
    let sensor_pos_right = vec2<i32>(
        i32(x + sim_size.agent_sensor_distance * cos(sensor_angle_right)),
        i32(y + sim_size.agent_sensor_distance * sin(sensor_angle_right))
    );
    let sensor_pos_center = vec2<i32>(
        i32(x + sim_size.agent_sensor_distance * cos(sensor_angle_center)),
        i32(y + sim_size.agent_sensor_distance * sin(sensor_angle_center))
    );

    // Sample the trail map at sensor positions
    var left_value: f32;
    if (sensor_pos_left.x < 0 || sensor_pos_left.x >= i32(sim_size.width) || sensor_pos_left.y < 0 || sensor_pos_left.y >= i32(sim_size.height)) {
        left_value = 0.0;
    } else {
        left_value = trail_map[sensor_pos_left.y * i32(sim_size.width) + sensor_pos_left.x];
    }
    var right_value: f32;
    if (sensor_pos_right.x < 0 || sensor_pos_right.x >= i32(sim_size.width) || sensor_pos_right.y < 0 || sensor_pos_right.y >= i32(sim_size.height)) {
        right_value = 0.0;
    } else {
        right_value = trail_map[sensor_pos_right.y * i32(sim_size.width) + sensor_pos_right.x];
    }
    var center_value: f32;
    if (sensor_pos_center.x < 0 || sensor_pos_center.x >= i32(sim_size.width) || sensor_pos_center.y < 0 || sensor_pos_center.y >= i32(sim_size.height)) {
        center_value = 0.0;
    } else {
        center_value = trail_map[sensor_pos_center.y * i32(sim_size.width) + sensor_pos_center.x];
    }

    // Update agent angle based on sensor values
    if (center_value > left_value && center_value > right_value) {
        // Continue straight
    } else if (left_value > right_value) {
        // Calculate shortest path to turn left
        let target_angle = angle - TAU;
        let angle_diff = target_angle - angle;
        angle += min(sim_size.agent_turn_speed, abs(angle_diff)) * sign(angle_diff);
    } else if (right_value > left_value) {
        // Calculate shortest path to turn right
        let target_angle = angle + TAU;
        let angle_diff = target_angle - angle;
        angle += min(sim_size.agent_turn_speed, abs(angle_diff)) * sign(angle_diff);
    } else {
        // If equal, do nothing
    }

    // Add jitter to angle
    let jitter_strength = sim_size.agent_jitter;
    // Create a more independent random value using agent index and time
    let random_val = fract(sin(f32(agent_index) * 12.9898 + x * 78.233 + y * 37.719) * 43758.5453);
    angle += (random_val * 2.0 - 1.0) * jitter_strength * 0.01;

    // Normalize angle to 0-2π range
    angle = angle % (2.0 * 3.14159265359);
    if (angle < 0.0) { angle = angle + 2.0 * 3.14159265359; }

    // Update agent position
    let move_dist = speed * TIME_STEP;
    x = x + move_dist * cos(angle);
    y = y + move_dist * sin(angle);

    // Wrap agent position to stay within bounds (toroidal)
    x = x % f32(sim_size.width);
    if (x < 0.0) { x = x + f32(sim_size.width); }
    y = y % f32(sim_size.height);
    if (y < 0.0) { y = y + f32(sim_size.height); }

    // Deposit trail
    let deposit_x = i32(x);
    let deposit_y = i32(y);
    if (deposit_x >= 0 && deposit_x < i32(sim_size.width) && deposit_y >= 0 && deposit_y < i32(sim_size.height)) {
        let idx = deposit_y * i32(sim_size.width) + deposit_x;
        trail_map[idx] = clamp(trail_map[idx] + sim_size.pheromone_deposition_amount, 0.0, 1.0);
    }

    // Update agent in the buffer
    agents[agent_index] = vec4<f32>(x, y, angle, speed);
}

// Add a new compute entry point for trail decay
@compute @workgroup_size(16, 16)
fn decay_trail(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    if (x >= i32(sim_size.width) || y >= i32(sim_size.height)) {
        return;
    }
    let idx = y * i32(sim_size.width) + x;
    let decay_factor = sim_size.decay_factor * 0.001;
    trail_map[idx] = max(trail_map[idx] - decay_factor, 0.0);
}

// Add a new compute entry point for diffusion
@compute @workgroup_size(16, 16)
fn diffuse_trail(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = i32(id.x);
    let y = i32(id.y);
    if (x >= i32(sim_size.width) || y >= i32(sim_size.height)) {
        return;
    }

    let idx = y * i32(sim_size.width) + x;
    let diffusion_rate = clamp(sim_size.diffusion_rate, 0.0, 1.0);
    
    // Box blur using 3x3 neighborhood
    var sum = 0.0;
    var count = 0.0;
    
    // Check all neighbors in 3x3 grid
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nx = x + dx;
            let ny = y + dy;
            
            // Skip if out of bounds
            if (nx < 0 || nx >= i32(sim_size.width) || ny < 0 || ny >= i32(sim_size.height)) {
                continue;
            }
            
            let sample_idx = ny * i32(sim_size.width) + nx;
            sum += trail_map[sample_idx];
            count += 1.0;
        }
    }
    
    // Apply diffusion if we have valid neighbors
    if (count > 0.0) {
        let avg_value = sum / count;
        trail_map[idx] = mix(trail_map[idx], avg_value, diffusion_rate);
    }
} 