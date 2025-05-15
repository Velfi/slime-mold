use crate::{
    DEFAULT_SETTINGS_FILE,
    agent::{Agent, AgentUpdate},
    errors::SlimeError,
    pheromones::Pheromones,
    rect::Rect,
    settings::Settings,
};
use colorgrad::preset::viridis;
use log::{error, info};
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

pub struct World {
    agents: Vec<Agent>,
    frame_time: f32,
    gradient: Arc<Box<dyn colorgrad::Gradient + Sync + Send>>,
    pheromones: Arc<RwLock<Pheromones>>,
    /// A toggle for rendering in color vs. black & white mode. Color mode has an FPS cost so we render in B&W by default
    black_and_white_mode: bool,
    settings: Settings,
    boundary_rect: Rect<u32>,
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    pub fn new(settings: Settings) -> Self {
        info!("generating {} agents", settings.agent_count);
        info!(
            r#"
AGENT_COUNT_MAX	{:?}
AGENT_JITTER	{:?}
AGENT_SPEED_	{:?}
AGENT_SPEED_	{:?}
AGENT_TURN_SPEED	{:?}
AGENT_POSSIBLE_STRT_HDGS	{:?}
DEPOSITION_AMOUN	{:?}
DECAY_FACTOR	{:?}
ENABLE_DYN_GRAD	{:?}
"#,
            settings.agent_count_maximum,
            settings.agent_jitter,
            settings.agent_speed_min,
            settings.agent_speed_max,
            settings.agent_turn_speed,
            settings.agent_possible_starting_headings,
            settings.agent_deposition_amount,
            settings.pheromone_decay_factor,
            settings.pheromone_enable_dynamic_gradient,
        );

        let agents: Vec<_> = (0..settings.agent_count)
            .map(|_| Agent::new_from_settings(&settings))
            .collect();

        let pheromones = Arc::new(RwLock::new(Pheromones::new(
            settings.window_width,
            settings.window_height,
            settings.pheromone_decay_factor,
            settings.pheromone_enable_dynamic_gradient,
            None,
        )));

        let gradient = Arc::new(Box::new(viridis()) as Box<dyn colorgrad::Gradient + Sync + Send>);

        let boundary_rect = Rect::new(0, 0, settings.window_width, settings.window_height);

        Self {
            agents,
            boundary_rect,
            frame_time: 0.0,
            gradient,
            pheromones,
            black_and_white_mode: true,
            settings,
        }
    }

    pub fn window_width(&self) -> u32 {
        self.settings.window_width
    }

    pub fn window_height(&self) -> u32 {
        self.settings.window_height
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.settings.window_width = width;
        self.settings.window_height = height;
        self.boundary_rect = Rect::new(0, 0, width, height);
        // Recreate pheromones with new dimensions
        // Assuming Pheromones::new takes width, height, decay_factor, enable_dynamic_gradient, and an optional static_gradient_generator
        // We need to decide how to handle the static_gradient_generator on resize. For now, passing None.
        self.pheromones = Arc::new(RwLock::new(Pheromones::new(
            width,
            height,
            self.settings.pheromone_decay_factor,
            self.settings.pheromone_enable_dynamic_gradient,
            None, // Or handle static gradient regeneration if needed
        )));
        info!("World resized to {}x{}", width, height);
    }

    pub fn reload_settings(&mut self) -> Result<(), SlimeError> {
        let new_settings = Settings::load_from_file(DEFAULT_SETTINGS_FILE)?;
        let agent_settings_changed = self.settings.did_agent_settings_change(&new_settings);
        let pheromone_settings_changed = self.settings.did_pheromone_settings_change(&new_settings);
        self.settings = new_settings;

        if agent_settings_changed {
            info!("agent settings have changed, running an update...");
            if let Err(e) = self.refresh_agents_from_settings() {
                error!(
                    "failed to update agents after a settings change with error: {}",
                    e
                );
            }
        }

        if pheromone_settings_changed {
            info!("pheromone settings have changed, running an update...");
            if let Err(e) = self.refresh_pheromones_from_settings() {
                error!(
                    "failed to update pheromones after a settings change with error: {}",
                    e
                );
            }
        }

        if self.agents.len() < self.settings.agent_count {
            (self.agents.len()..self.settings.agent_count)
                .for_each(|_| self.agents.push(Agent::new_from_settings(&self.settings)));
        }

        Ok(())
    }

    pub fn set_frame_time(&mut self, frame_time: f32) {
        self.frame_time = frame_time;
    }

    pub fn toggle_black_and_white_mode(&mut self) {
        self.black_and_white_mode = !self.black_and_white_mode
    }

    pub fn toggle_dynamic_gradient(&mut self) {
        self.settings.pheromone_enable_dynamic_gradient =
            !self.settings.pheromone_enable_dynamic_gradient;

        let _ = self.refresh_pheromones_from_settings();
    }

    fn refresh_agents_from_settings(&mut self) -> Result<(), SlimeError> {
        let agent_update = AgentUpdate {
            jitter: Some(self.settings.agent_jitter),
            rotation_speed: Some(self.settings.agent_turn_speed),
            deposition_amount: Some(self.settings.agent_deposition_amount),
            ..Default::default()
        };

        let move_speed_range = self.settings.agent_speed_min..self.settings.agent_speed_max;

        self.agents.iter_mut().for_each(|agent| {
            agent.apply_update(&agent_update);
            agent.set_new_random_move_speed_in_range(move_speed_range.clone());
        });

        Ok(())
    }

    fn refresh_pheromones_from_settings(&mut self) -> Result<(), SlimeError> {
        let mut pheromones = self
            .pheromones
            .write()
            .map_err(|e| SlimeError::ThreadSafety(e.to_string()))?;

        if self.settings.pheromone_enable_dynamic_gradient {
            pheromones.enable_dynamic_gradient();
        } else {
            pheromones.disable_dynamic_gradient();
        }

        pheromones.set_decay_factor(self.settings.pheromone_decay_factor);

        Ok(())
    }

    /// Update the `World` internal state; bounce the box around the screen.
    pub fn update(&mut self) {
        let boundary_rect = &self.boundary_rect;
        let pheromones = &self.pheromones;
        let agents = &mut self.agents;
        // More stuff should be affected by delta_t in order to make the simulation run at the same speed
        // regardless of how fast the program is actually running. Right now it just affects agent speed.
        let delta_t = self.frame_time;

        agents.par_iter_mut().for_each(|agent| {
            let pheromones = pheromones
                .read()
                .expect("reading pheromones during agent update");

            agent.update(&pheromones, delta_t, boundary_rect)
        });

        let pheromones = &mut self.pheromones;
        let agents = &self.agents;

        pheromones
            .write()
            .expect("couldn't get mut ref to pheromones for deposit step")
            .deposit(agents);

        pheromones
            .write()
            .expect("couldn't get mut ref to pheromones for diffuse step")
            .diffuse();
        pheromones
            .write()
            .expect("couldn't get mut ref to pheromones for decay step")
            .decay();

        if self.agents.len() > self.settings.agent_count_maximum {
            self.agents.truncate(self.settings.agent_count_maximum)
        }
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    /// Assumes that pheromone grid and pixel FB have same dimensions
    pub fn draw(&self) -> Vec<u8> {
        let num_rows = self.settings.window_height as usize;
        let num_cols = self.settings.window_width as usize;
        let mut frame_data = vec![0u8; num_rows * num_cols * 4]; // RGBA8

        let pixel_iter = frame_data.par_chunks_exact_mut(4);
        let gradient = Arc::clone(&self.gradient);

        let pheromones_guard = self
            .pheromones
            .read()
            .expect("couldn't get lock on pheromones for draw");

        // Access the underlying DMatrix
        // The Pheromones::iter() method used self.grid.a().iter(), which is column-major.
        // We need to iterate self.grid.a() in row-major order.
        let matrix = pheromones_guard.get_current_grid(); // This is &DMatrix<f32>

        // Uncomment to enable rendering of the static gradient for debugging
        // let matrix = pheromones_guard.static_gradient().expect("Static gradient not available for drawing");

        let mut pheromones_row_major: Vec<f32> = Vec::with_capacity(num_rows * num_cols);

        for r in 0..num_rows {
            for c in 0..num_cols {
                pheromones_row_major.push(matrix[(r, c)]);
            }
        }

        pixel_iter
            .zip_eq(pheromones_row_major.par_iter()) // Use the new row-major Vec
            .for_each(|(pixel, pheromone_value)| {
                // clamp to renderable range
                // map cell pheromone values to rgba pixels
                if self.black_and_white_mode {
                    // Ensure all operations are on f32 before final cast
                    let val_f32: f32 = *pheromone_value * 255.0f32;
                    let rounded_f32: f32 = val_f32.round();
                    let clamped_f32: f32 = rounded_f32.clamp(0.0f32, 255.0f32);
                    let val_u8: u8 = clamped_f32 as u8;
                    pixel.copy_from_slice(&[val_u8, val_u8, val_u8, 0xff]);
                } else {
                    // Ensure pheromone_value is dereferenced and clamped as f32 before casting to f64
                    let clamped_pheromone = (*pheromone_value).clamp(0.0f32, 1.0f32);
                    let rgba = gradient.at(clamped_pheromone).to_rgba8(); // colorgrad::Gradient::at expects f64
                    pixel.copy_from_slice(&rgba);
                }
            });

        frame_data
    }
}
