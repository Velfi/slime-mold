use criterion::{criterion_group, criterion_main, Criterion};
use slime_mold::{
    settings::Settings,
    gpu_state::GpuState,
    lut_manager::LutManager,
};
use winit::event_loop::{EventLoop, ActiveEventLoop};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::window::WindowId;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex, OnceLock};

struct BenchmarkApp {
    gpu_state: Rc<RefCell<Option<GpuState>>>,
}

impl ApplicationHandler for BenchmarkApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu_state.borrow().is_none() {
            let gpu_state = pollster::block_on(GpuState::new(
                event_loop,
                1600,
                900,
                false,
                1_000_000,
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

fn benchmark_agent_update(c: &mut Criterion) {
    let gpu_state_arc = get_shared_gpu_state();
    let mut group = c.benchmark_group("agent_update");
    
    group.bench_function("update_agents", |b| {
        b.iter(|| {
            let gpu_state = gpu_state_arc.lock().unwrap();
            let mut encoder = gpu_state.create_command_encoder();
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Agent Update Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&gpu_state.pipeline_manager().compute_pipeline);
                cpass.set_bind_group(0, &gpu_state.bind_group_manager().compute_bind_group, &[]);
                cpass.dispatch_workgroups(
                    gpu_state.workgroup_config().workgroups_1d(1_000_000u32), 
                    1, 
                    1
                );
            }
            gpu_state.submit(encoder.finish());
        });
    });
    group.finish();
}

fn benchmark_trail_decay(c: &mut Criterion) {
    let gpu_state_arc = get_shared_gpu_state();
    let mut group = c.benchmark_group("trail_decay");
    
    group.bench_function("decay_trails", |b| {
        b.iter(|| {
            let gpu_state = gpu_state_arc.lock().unwrap();
            let mut encoder = gpu_state.create_command_encoder();
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Trail Decay Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&gpu_state.pipeline_manager().decay_pipeline);
                cpass.set_bind_group(0, &gpu_state.bind_group_manager().compute_bind_group, &[]);
                cpass.dispatch_workgroups(
                    gpu_state.workgroup_config().workgroups_1d(
                        gpu_state.config().width * gpu_state.config().height
                    ),
                    1,
                    1,
                );
            }
            gpu_state.submit(encoder.finish());
        });
    });
    group.finish();
}

fn benchmark_trail_diffusion(c: &mut Criterion) {
    let gpu_state_arc = get_shared_gpu_state();
    let mut group = c.benchmark_group("trail_diffusion");
    
    group.bench_function("diffuse_trails", |b| {
        b.iter(|| {
            let gpu_state = gpu_state_arc.lock().unwrap();
            let mut encoder = gpu_state.create_command_encoder();
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Trail Diffusion Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&gpu_state.pipeline_manager().diffuse_pipeline);
                cpass.set_bind_group(0, &gpu_state.bind_group_manager().compute_bind_group, &[]);
                cpass.dispatch_workgroups(
                    gpu_state.workgroup_config().workgroups_1d(
                        gpu_state.config().width * gpu_state.config().height
                    ),
                    1,
                    1,
                );
            }
            gpu_state.submit(encoder.finish());
        });
    });
    group.finish();
}

fn benchmark_display_update(c: &mut Criterion) {
    let gpu_state_arc = get_shared_gpu_state();
    let mut group = c.benchmark_group("display_update");
    
    group.bench_function("update_display", |b| {
        b.iter(|| {
            let gpu_state = gpu_state_arc.lock().unwrap();
            let mut encoder = gpu_state.create_command_encoder();
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Display Update Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&gpu_state.pipeline_manager().display_pipeline);
                cpass.set_bind_group(0, &gpu_state.bind_group_manager().display_bind_group, &[]);
                let (x_groups, y_groups) = gpu_state.workgroup_config().workgroups_2d(
                    gpu_state.config().width, 
                    gpu_state.config().height
                );
                cpass.dispatch_workgroups(x_groups, y_groups, 1);
            }
            gpu_state.submit(encoder.finish());
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_agent_update,
    benchmark_trail_decay,
    benchmark_trail_diffusion,
    benchmark_display_update
);
criterion_main!(benches); 