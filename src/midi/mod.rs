mod interface;
mod summit;

use crate::agent::AgentUpdate;
pub use interface::MidiInterface;
pub use midly::MidiMessage;
pub use summit::SummitController;

// Notes on my synth are in the 36 to 96 range by default
// vel ranges from 0 to 127
fn new_agent_from_midi(key: u8, vel: u8) -> Agent {
    let mut rng: StdRng = SeedableRng::from_entropy();
    let move_speed = rng.gen_range(AGENT_SPEED_MIN..AGENT_SPEED_MAX);
    let location = Point2::new(
        map_range(key as f64, 36.0, 96.0, 0.0, WIDTH as f64),
        map_range(vel as f64, 0.0, 127.0, 0.0, HEIGHT as f64),
    );

    let heading = rng.gen_range(AGENT_POSSIBLE_STARTING_HEADINGS);

    Agent::builder()
        .location(location)
        .heading(heading)
        .move_speed(move_speed)
        .jitter(AGENT_JITTER)
        .deposition_amount(DEPOSITION_AMOUNT)
        .rotation_speed(AGENT_TURN_SPEED)
        .rng(rng)
        .build()
}

fn handle_controller_input(world: &mut World, controller: u8, value: u8) {
    match controller {
        // Noise Level
        27 => {
            let jitter = Some(map_range(value as f64, 0.0, 127.0, 2.0, 40.0));
            let agent_update = AgentUpdate {
                jitter,
                ..Default::default()
            };

            world
                .agents
                .iter_mut()
                .for_each(|agent| agent.apply_update(&agent_update));
        }
        // Filter Freq
        29 => {
            let deposition_amount = Some(map_range(value as f64, 0.0, 127.0, 0.2, 4.0));
            let agent_update = AgentUpdate {
                deposition_amount,
                ..Default::default()
            };

            world
                .agents
                .iter_mut()
                .for_each(|agent| agent.apply_update(&agent_update));
        }
        _ => (),
    };
}
