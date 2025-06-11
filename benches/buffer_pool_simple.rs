use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use slime_mold::buffer_pool::BufferPool;
use std::hint::black_box;
use wgpu::{Device, Queue, Instance, BufferUsages};
use pollster;

async fn create_device() -> (Device, Queue) {
    let instance = Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("No compatible adapter found");

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                memory_hints: wgpu::MemoryHints::default(),
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to create device")
}

fn benchmark_buffer_pool_vs_direct(c: &mut Criterion) {
    let (device, _queue) = pollster::block_on(create_device());
    let mut group = c.benchmark_group("buffer_allocation_comparison");

    // Test various buffer sizes that are commonly used in the simulation
    let test_sizes = [
        ("small", 4096),     // 4KB
        ("medium", 65536),   // 64KB  
        ("large", 1048576),  // 1MB
        ("xlarge", 16777216), // 16MB
    ];

    for (size_name, size) in test_sizes {
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        // Benchmark direct allocation
        group.bench_function(format!("direct_{}", size_name), |b| {
            b.iter_batched(
                || (),
                |_| {
                    // Create and immediately drop buffers (simulating heavy allocation/deallocation)
                    let mut buffers = Vec::new();
                    for _ in 0..10 {
                        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Direct Benchmark Buffer"),
                            size,
                            usage,
                            mapped_at_creation: false,
                        });
                        buffers.push(buffer);
                    }
                    black_box(buffers);
                    // Buffers are dropped here
                },
                BatchSize::SmallInput,
            );
        });

        // Benchmark pooled allocation
        group.bench_function(format!("pooled_{}", size_name), |b| {
            b.iter_batched(
                || BufferPool::new(),
                |mut pool| {
                    // Get and return buffers from/to pool
                    let mut buffers = Vec::new();
                    for _ in 0..10 {
                        let buffer = pool.get_buffer(
                            &device,
                            Some("Pooled Benchmark Buffer"),
                            size,
                            usage,
                        );
                        buffers.push((buffer, size));
                    }
                    
                    black_box(&buffers);
                    
                    // Return all buffers to pool
                    for (buffer, size) in buffers {
                        pool.return_buffer(buffer, size, usage);
                    }
                },
                BatchSize::SmallInput,
            );
        });

        // Benchmark repeated reuse (this should show the most benefit for pooling)
        group.bench_function(format!("pooled_reuse_{}", size_name), |b| {
            let mut pool = BufferPool::new();
            // Pre-populate the pool
            let mut initial_buffers = Vec::new();
            for _ in 0..10 {
                let buffer = pool.get_buffer(&device, Some("Initial Buffer"), size, usage);
                initial_buffers.push((buffer, size));
            }
            for (buffer, size) in initial_buffers {
                pool.return_buffer(buffer, size, usage);
            }

            b.iter(|| {
                // Now all allocations should come from the pool
                let mut buffers = Vec::new();
                for _ in 0..10 {
                    let buffer = pool.get_buffer(&device, Some("Reuse Buffer"), size, usage);
                    buffers.push((buffer, size));
                }
                
                black_box(&buffers);
                
                // Return all buffers to pool
                for (buffer, size) in buffers {
                    pool.return_buffer(buffer, size, usage);
                }
            });
        });
    }

    group.finish();
}

fn benchmark_resize_simulation(c: &mut Criterion) {
    let (device, _queue) = pollster::block_on(create_device());
    let mut group = c.benchmark_group("resize_simulation");

    // Simulate the resize pattern that happens when user changes settings
    let resize_pattern = [
        (10_000, 800, 600),   // Initial state
        (15_000, 800, 600),   // Increase agents
        (15_000, 1200, 800),  // Increase window size
        (8_000, 1200, 800),   // Decrease agents
        (8_000, 800, 600),    // Decrease window size
        (20_000, 1600, 900),  // Large increase
    ];

    group.bench_function("direct_resize_pattern", |b| {
        b.iter(|| {
            let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
            
            for &(agent_count, width, height) in &resize_pattern {
                let agent_size = (agent_count * 4 * std::mem::size_of::<f32>()) as u64;
                let trail_size = (width * height * std::mem::size_of::<u32>()) as u64;
                let gradient_size = trail_size;

                // Create new buffers (simulate old buffer destruction)
                let agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Agent Buffer"),
                    size: agent_size,
                    usage,
                    mapped_at_creation: false,
                });

                let trail_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Trail Buffer"),
                    size: trail_size,
                    usage,
                    mapped_at_creation: false,
                });

                let gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Gradient Buffer"),
                    size: gradient_size,
                    usage,
                    mapped_at_creation: false,
                });

                black_box((&agent_buffer, &trail_buffer, &gradient_buffer));
                // Buffers will be dropped, simulating the old behavior
            }
        });
    });

    group.bench_function("pooled_resize_pattern", |b| {
        b.iter(|| {
            let mut pool = BufferPool::new();
            let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
            let mut old_buffers = Vec::new();
            
            for &(agent_count, width, height) in &resize_pattern {
                let agent_size = (agent_count * 4 * std::mem::size_of::<f32>()) as u64;
                let trail_size = (width * height * std::mem::size_of::<u32>()) as u64;
                let gradient_size = trail_size;

                // Return old buffers to pool (except first iteration)
                for (buffer, size) in old_buffers.drain(..) {
                    pool.return_buffer(buffer, size, usage);
                }

                // Get new buffers from pool
                let agent_buffer = pool.get_buffer(&device, Some("Agent Buffer"), agent_size, usage);
                let trail_buffer = pool.get_buffer(&device, Some("Trail Buffer"), trail_size, usage);
                let gradient_buffer = pool.get_buffer(&device, Some("Gradient Buffer"), gradient_size, usage);

                black_box((&agent_buffer, &trail_buffer, &gradient_buffer));

                // Store for next iteration
                old_buffers.push((agent_buffer, agent_size));
                old_buffers.push((trail_buffer, trail_size));
                old_buffers.push((gradient_buffer, gradient_size));
            }

            // Return final buffers
            for (buffer, size) in old_buffers {
                pool.return_buffer(buffer, size, usage);
            }
        });
    });

    group.finish();
}

fn benchmark_pool_overhead(c: &mut Criterion) {
    let (device, _queue) = pollster::block_on(create_device());
    let mut group = c.benchmark_group("pool_overhead");

    group.bench_function("pool_get_return_cycle", |b| {
        let mut pool = BufferPool::new();
        let size = 65536u64;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        b.iter(|| {
            let buffer = pool.get_buffer(&device, Some("Cycle Buffer"), size, usage);
            black_box(&buffer);
            pool.return_buffer(buffer, size, usage);
        });
    });

    group.bench_function("pool_stats_collection", |b| {
        let mut pool = BufferPool::new();
        let size = 65536u64;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        // Add some buffers to the pool
        let mut buffers = Vec::new();
        for _ in 0..5 {
            let buffer = pool.get_buffer(&device, Some("Stats Buffer"), size, usage);
            buffers.push((buffer, size));
        }
        for (buffer, size) in buffers {
            pool.return_buffer(buffer, size, usage);
        }

        b.iter(|| {
            let stats = pool.memory_stats();
            black_box(stats);
        });
    });

    group.finish();
}

criterion_group!(
    buffer_pool_benches,
    benchmark_buffer_pool_vs_direct,
    benchmark_resize_simulation,
    benchmark_pool_overhead
);
criterion_main!(buffer_pool_benches); 