use criterion::{criterion_group, criterion_main, Criterion, black_box};
use slime_mold::{
    settings::Settings,
    gpu_state::GpuState,
    lut_manager::LutManager,
    buffer_pool::BufferPool,
};
use winit::event_loop::{EventLoop, ActiveEventLoop};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::window::WindowId;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::{Device, BufferUsages};

struct BenchmarkApp {
    gpu_state: Rc<RefCell<Option<GpuState>>>,
}

impl ApplicationHandler for BenchmarkApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu_state.borrow().is_none() {
            let gpu_state = pollster::block_on(GpuState::new(
                event_loop,
                800,
                600,
                false,
                10_000,
                &Settings::default(),
                &LutManager::new(),
                &vec!["MATPLOTLIB_bone_r".to_string()],
                0,
                false,
            )).unwrap();
            *self.gpu_state.borrow_mut() = Some(gpu_state);
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }
    }
}

static GPU_STATE: OnceLock<Arc<Mutex<GpuState>>> = OnceLock::new();

fn get_shared_gpu_state() -> Arc<Mutex<GpuState>> {
    GPU_STATE.get_or_init(|| {
        let event_loop = EventLoop::new().unwrap();
        let gpu_state_rc = Rc::new(RefCell::new(None));
        let mut app = BenchmarkApp { 
            gpu_state: gpu_state_rc.clone()
        };
        
        let _ = event_loop.run_app(&mut app);
        
        // Extract the GPU state
        let gpu_state = gpu_state_rc.borrow_mut().take().unwrap();
        Arc::new(Mutex::new(gpu_state))
    }).clone()
}

fn benchmark_buffer_allocation_direct(c: &mut Criterion) {
    let gpu_state_arc = get_shared_gpu_state();
    let mut group = c.benchmark_group("buffer_allocation");
    
    group.bench_function("direct_allocation", |b| {
        b.iter(|| {
            let gpu_state = gpu_state_arc.lock().unwrap();
            let device = &gpu_state.device;
            
            // Simulate creating and dropping buffers of various sizes
            let sizes = [1024, 4096, 16384, 65536, 262144]; // Various buffer sizes
            let mut buffers = Vec::new();
            
            for &size in &sizes {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Benchmark Buffer"),
                    size,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                buffers.push(buffer);
            }
            
            // Simulate using the buffers
            black_box(&buffers);
            
            // Buffers will be dropped here
        });
    });
    
    group.finish();
}

fn benchmark_buffer_allocation_pooled(c: &mut Criterion) {
    let gpu_state_arc = get_shared_gpu_state();
    let mut group = c.benchmark_group("buffer_allocation");
    
    group.bench_function("pooled_allocation", |b| {
        b.iter(|| {
            let gpu_state = gpu_state_arc.lock().unwrap();
            let device = &gpu_state.device;
            let mut buffer_pool = BufferPool::new();
            
            // Simulate creating and returning buffers of various sizes
            let sizes = [1024, 4096, 16384, 65536, 262144]; // Various buffer sizes
            let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
            let mut buffers = Vec::new();
            
            for &size in &sizes {
                let buffer = buffer_pool.get_buffer(
                    device,
                    Some("Benchmark Buffer"),
                    size,
                    usage,
                );
                buffers.push((buffer, size));
            }
            
            // Simulate using the buffers
            black_box(&buffers);
            
            // Return buffers to pool
            for (buffer, size) in buffers {
                buffer_pool.return_buffer(buffer, size, usage);
            }
        });
    });
    
    group.finish();
}

fn benchmark_buffer_reallocation_pattern(c: &mut Criterion) {
    let gpu_state_arc = get_shared_gpu_state();
    let mut group = c.benchmark_group("buffer_reallocation");
    
    group.bench_function("direct_reallocation_pattern", |b| {
        b.iter(|| {
            let gpu_state = gpu_state_arc.lock().unwrap();
            let device = &gpu_state.device;
            
            // Simulate the pattern of reallocating buffers during resize operations
            // This mimics what happens when user changes agent count or window size
            let iterations = 5;
            
            for i in 0..iterations {
                let agent_count = 10_000 + i * 5_000;
                let trail_map_size = 800 * 600;
                
                // Create agent buffer
                let agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Agent Buffer"),
                    size: (agent_count * 4 * std::mem::size_of::<f32>()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                
                // Create trail map buffer
                let trail_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Trail Buffer"),
                    size: (trail_map_size * std::mem::size_of::<f32>()) as u64,
                    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                
                black_box((&agent_buffer, &trail_buffer));
                // Buffers dropped at end of scope
            }
        });
    });
    
    group.bench_function("pooled_reallocation_pattern", |b| {
        b.iter(|| {
            let gpu_state = gpu_state_arc.lock().unwrap();
            let device = &gpu_state.device;
            let mut buffer_pool = BufferPool::new();
            let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
            
            // Simulate the pattern of reallocating buffers during resize operations
            let iterations = 5;
            let mut agent_buffers = Vec::new();
            let mut trail_buffers = Vec::new();
            
            for i in 0..iterations {
                let agent_count = 10_000 + i * 5_000;
                let trail_map_size = 800 * 600;
                
                let agent_size = (agent_count * 4 * std::mem::size_of::<f32>()) as u64;
                let trail_size = (trail_map_size * std::mem::size_of::<f32>()) as u64;
                
                // Get buffers from pool
                let agent_buffer = buffer_pool.get_buffer(
                    device,
                    Some("Agent Buffer"),
                    agent_size,
                    usage,
                );
                
                let trail_buffer = buffer_pool.get_buffer(
                    device,
                    Some("Trail Buffer"),
                    trail_size,
                    usage,
                );
                
                black_box((&agent_buffer, &trail_buffer));
                
                // Store for later return to pool
                agent_buffers.push((agent_buffer, agent_size));
                trail_buffers.push((trail_buffer, trail_size));
            }
            
            // Return all buffers to pool
            for (buffer, size) in agent_buffers {
                buffer_pool.return_buffer(buffer, size, usage);
            }
            for (buffer, size) in trail_buffers {
                buffer_pool.return_buffer(buffer, size, usage);
            }
        });
    });
    
    group.finish();
}

fn benchmark_resize_operations(c: &mut Criterion) {
    let gpu_state_arc = get_shared_gpu_state();
    let mut group = c.benchmark_group("resize_operations");
    
    group.bench_function("resize_buffers_with_pool", |b| {
        b.iter(|| {
            let mut gpu_state = gpu_state_arc.lock().unwrap();
            let settings = Settings::default();
            
            // Simulate multiple resize operations
            let agent_counts = [5_000, 10_000, 15_000, 8_000, 12_000];
            
            for &agent_count in &agent_counts {
                gpu_state.resize_buffers(agent_count, &settings);
                black_box(&gpu_state);
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    buffer_benches,
    benchmark_buffer_allocation_direct,
    benchmark_buffer_allocation_pooled,
    benchmark_buffer_reallocation_pattern,
    benchmark_resize_operations
);
criterion_main!(buffer_benches); 