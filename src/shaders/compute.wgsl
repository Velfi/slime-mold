// Compute shader for Physarum simulation
// Each agent is represented by a vec4<f32>: x, y, angle, speed

const TAU: f32 = 6.28318530718; // 2π

struct SimSizeUniform {
    width: u32,
    height: u32,
    decay_rate: f32,
    agent_jitter: f32,
    agent_speed_min: f32,
    agent_speed_max: f32,
    agent_turn_rate: f32,
    agent_sensor_angle: f32,
    agent_sensor_distance: f32,
    diffusion_rate: f32,
    pheromone_deposition_rate: f32,
    gradient_enabled: u32,
    gradient_type: u32,
    gradient_strength: f32,
    gradient_center_x: f32,
    gradient_center_y: f32,
    gradient_size: f32,
    gradient_angle: f32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read_write> agents: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> trail_map: array<f32>;

@group(0) @binding(2)
var<uniform> sim_size: SimSizeUniform;

@group(0) @binding(3)
var<storage, read> gradient_map: array<f32>;

// Helper function for bilinear interpolation
fn sample_trail_map(pos: vec2<f32>) -> f32 {
    let width = i32(sim_size.width);
    let height = i32(sim_size.height);

    let x0 = ((i32(floor(pos.x)) % width) + width) % width;
    let y0 = ((i32(floor(pos.y)) % height) + height) % height;
    let x1 = (x0 + 1) % width;
    let y1 = (y0 + 1) % height;

    let dx = pos.x - f32(i32(floor(pos.x)));
    let dy = pos.y - f32(i32(floor(pos.y)));

    let v00 = trail_map[y0 * width + x0];
    let v10 = trail_map[y0 * width + x1];
    let v01 = trail_map[y1 * width + x0];
    let v11 = trail_map[y1 * width + x1];

    let v0 = mix(v00, v10, dx);
    let v1 = mix(v01, v11, dx);
    return mix(v0, v1, dy);
}

// Helper function to sample gradient map
fn sample_gradient_map(pos: vec2<f32>) -> f32 {
    let width = i32(sim_size.width);
    let height = i32(sim_size.height);

    let x0 = ((i32(floor(pos.x)) % width) + width) % width;
    let y0 = ((i32(floor(pos.y)) % height) + height) % height;
    let x1 = (x0 + 1) % width;
    let y1 = (y0 + 1) % height;

    let dx = pos.x - f32(i32(floor(pos.x)));
    let dy = pos.y - f32(i32(floor(pos.y)));

    let v00 = gradient_map[y0 * width + x0];
    let v10 = gradient_map[y0 * width + x1];
    let v01 = gradient_map[y1 * width + x0];
    let v11 = gradient_map[y1 * width + x1];

    let v0 = mix(v00, v10, dx);
    let v1 = mix(v01, v11, dx);
    return mix(v0, v1, dy);
}

// Combined function to sample both trail and gradient
fn sample_combined_map(pos: vec2<f32>) -> f32 {
    let trail_value = sample_trail_map(pos);
    var gradient_value: f32;
    if (sim_size.gradient_enabled == 1u) {
        gradient_value = sample_gradient_map(pos);
    } else {
        gradient_value = 0.0;
    }
    return trail_value + gradient_value;
}

// Parameters for the simulation (now mostly from uniform)
const TIME_STEP: f32 = 0.016; // Affects how far agents move per frame based on their speed

// Add grid cell size constant
const GRID_CELL_SIZE: f32 = 10.0;  // Should be larger than repulsion radius

// Constants for spatial partitioning
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const CELL_SIZE: f32 = 20.0;  // Size of each cell in the spatial grid

// Shared memory for storing local agent positions
var<workgroup> local_agents: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn update_agents(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let agent_index = id.x;
    // Remove the agent_count check since we'll handle this with dispatch workgroup size
    // if (agent_index >= u32(sim_size.agent_count)) {
    //     return;
    // }

    // Get agent data
    let agent = agents[agent_index];
    var x = agent.x;
    var y = agent.y;
    var angle = agent.z;
    var speed = agent.w;

    // Sample trail map at sensor positions
    let sensor_distance = sim_size.agent_sensor_distance;
    let sensor_angle = sim_size.agent_sensor_angle;
    
    // Calculate sensor positions
    let left_angle = angle - sensor_angle;
    let right_angle = angle + sensor_angle;
    
    let left_pos = vec2<f32>(
        x + cos(left_angle) * sensor_distance,
        y + sin(left_angle) * sensor_distance
    );
    let right_pos = vec2<f32>(
        x + cos(right_angle) * sensor_distance,
        y + sin(right_angle) * sensor_distance
    );
    
    // Sample combined trail + gradient maps at sensor positions
    let left_value = sample_combined_map(left_pos);
    let right_value = sample_combined_map(right_pos);
    
    // Update angle based on sensor readings
    if (left_value > right_value) {
        // Calculate shortest path to turn left
        let target_angle = angle - TAU;
        let angle_diff = target_angle - angle;
        angle += min(sim_size.agent_turn_rate, abs(angle_diff)) * sign(angle_diff);
    } else if (right_value > left_value) {
        // Calculate shortest path to turn right
        let target_angle = angle + TAU;
        let angle_diff = target_angle - angle;
        angle += min(sim_size.agent_turn_rate, abs(angle_diff)) * sign(angle_diff);
    } else {
        // If equal, do nothing
    }

    // Update agent position
    let move_dist = speed * TIME_STEP;
    x = x + move_dist * cos(angle);
    y = y + move_dist * sin(angle);

    // Calculate repulsion using workgroup-based approach
    var repulsion_force = vec2<f32>(0.0, 0.0);
    let repulsion_radius = 8.0;  // Increased from 5.0 to 8.0
    let repulsion_strength = 2.0;  // Fixed constant value since agent_repulsion_strength was removed

    // Calculate cell coordinates
    let cell_x = i32(x / CELL_SIZE);
    let cell_y = i32(y / CELL_SIZE);

    // Store agent in shared memory
    let local_index = i32(local_id.x);
    local_agents[local_index] = vec4<f32>(x, y, angle, speed);
    workgroupBarrier();

    // Check repulsion against agents in the same workgroup
    for (var i = 0; i < i32(WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y); i++) {
        if (i == local_index) {
            continue;  // Skip self
        }

        let other_agent = local_agents[i];
        let dx = other_agent.x - x;
        let dy = other_agent.y - y;
        let dist_sq = dx * dx + dy * dy;

        if (dist_sq < repulsion_radius * repulsion_radius) {
            let dist = sqrt(dist_sq);
            // Modified force calculation to be stronger at close range
            let force = repulsion_strength * pow(1.0 - dist / repulsion_radius, 2.0);
            repulsion_force += vec2<f32>(-dx / dist, -dy / dist) * force;
        }
    }

    // Apply repulsion force
    x += repulsion_force.x;
    y += repulsion_force.y;

    // Apply jitter
    let jitter_strength = sim_size.agent_jitter;
    let random_x = fract(sin(f32(agent_index)) * 1e4);
    let random_y = fract(sin(f32(agent_index) + 1.0) * 1e4);
    x += (random_x * 2.0 - 1.0) * jitter_strength;
    y += (random_y * 2.0 - 1.0) * jitter_strength;

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
        trail_map[idx] = clamp(trail_map[idx] + sim_size.pheromone_deposition_rate, 0.0, 1.0);
    }

    // Update agent in the buffer
    agents[agent_index] = vec4<f32>(x, y, angle, speed);
}

// Add a new compute entry point for trail decay
@compute @workgroup_size(256)
fn decay_trail(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total_size = sim_size.width * sim_size.height;
    if (idx >= total_size) {
        return;
    }

    // Toroidal wrapping for index
    let x = idx % sim_size.width;
    let y = idx / sim_size.width;
    let wrapped_x = (x + sim_size.width) % sim_size.width;
    let wrapped_y = (y + sim_size.height) % sim_size.height;
    let wrapped_idx = wrapped_y * sim_size.width + wrapped_x;
    
    // Apply decay rate (now 1.0 is normal value)
    let decay_rate = sim_size.decay_rate * 0.001; // 1.0 = 0.1% decay per frame
    trail_map[wrapped_idx] = max(0.0, trail_map[wrapped_idx] - decay_rate);
}

// Add a new compute entry point for diffusion
@compute @workgroup_size(256)
fn diffuse_trail(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total_size = sim_size.width * sim_size.height;
    if (idx >= total_size) {
        return;
    }

    let x = idx % sim_size.width;
    let y = idx / sim_size.width;
    
    // Get neighboring values with toroidal wrapping
    let x_prev = (x + sim_size.width - 1) % sim_size.width;
    let x_next = (x + 1) % sim_size.width;
    let y_prev = (y + sim_size.height - 1) % sim_size.height;
    let y_next = (y + 1) % sim_size.height;
    
    let center = trail_map[y * sim_size.width + x];
    let left = trail_map[y * sim_size.width + x_prev];
    let right = trail_map[y * sim_size.width + x_next];
    let up = trail_map[y_prev * sim_size.width + x];
    let down = trail_map[y_next * sim_size.width + x];
    
    // Simple diffusion: average of neighbors
    let diffusion_rate = sim_size.diffusion_rate;
    let new_value = center * (1.0 - diffusion_rate) + 
                   (left + right + up + down) * (diffusion_rate * 0.25);
    
    trail_map[y * sim_size.width + x] = new_value;
}

@compute @workgroup_size(256)
fn update_agent_speeds(@builtin(global_invocation_id) id: vec3<u32>) {
    let agent_index = id.x;
    
    // Get current agent data
    let agent = agents[agent_index];
    let x = agent.x;
    let y = agent.y;
    let angle = agent.z;
    
    // Generate new random speed within the current range
    let random_speed = fract(sin(f32(agent_index) * 12.9898 + 78.233) * 43758.5453);
    let speed_range = sim_size.agent_speed_max - sim_size.agent_speed_min;
    let new_speed = sim_size.agent_speed_min + random_speed * speed_range;
    
    // Update agent with new speed
    agents[agent_index] = vec4<f32>(x, y, angle, new_speed);
} 