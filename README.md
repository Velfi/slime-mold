# Slime Mold

A GPU-accelerated slime mold simulation written in Rust using WGPU. This project simulates the emergent behavior of slime mold colonies using compute shaders. It supports millions of agents on modern hardware.

## Features

- Real-time GPU-accelerated simulation using WGPU
- Interactive parameter customization for slime behavior
- Multiple presets for different simulation patterns
- Custom LUT (Look-Up Table) support for color visualization

## Quickstart

Run the simulation with `cargo`:

```sh
cargo run --release
```

## Controls

The simulation can be controlled using the following keyboard shortcuts:
- Press `/` to toggle help text
- Press `P` to cycle presets (Shift+P for reverse)
- Press `C` to clear trail map
- Press `G` to cycle LUTs (Shift+G for reverse)
- Hold `T` and use arrow keys to adjust turn speed
- Hold `J` and use arrow keys to adjust jitter
- Hold `S` and use arrow keys to adjust speed range
- Hold `A` and use arrow keys to adjust sensor angle
- Hold `D` and use arrow keys to adjust sensor distance
- Hold `R` and use arrow keys to adjust pheromone deposition amount
- Hold `V` and use arrow keys to adjust pheromone decay factor
- Hold `B` and use arrow keys to adjust pheromone diffusion rate
- Hold `N` and use arrow keys to adjust agent count (1M/100K increments)
- Hold `Shift` with any of the above for fine adjustments

## License

This project is open source and available under the MIT License.
