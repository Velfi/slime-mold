use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;

use slime_mold::app::App;
fn main() {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
