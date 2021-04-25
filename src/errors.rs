use std::sync::RwLockWriteGuard;

#[cfg(feature = "midi")]
use midir::MidiInput;
use thiserror::Error;

// TODO so many of these just wrap; maybe you should try using anyhow instead?
#[derive(Debug, Error)]
pub enum SlimeError {
    #[cfg(feature = "midi")]
    #[error("Midir encountered an issue: {0}")]
    MidirConnection(#[from] midir::ConnectError<MidiInput>),
    #[cfg(feature = "midi")]
    #[error("Couldn't detect any MIDI devices")]
    NoMidiDevicesDetected,
    #[cfg(feature = "midi")]
    #[error("Couldn't find a MIDI device with the name '{0}', please check your spelling")]
    NoMidiDevicesWithName(String),
    #[cfg(feature = "midi")]
    #[error("The MIDI port you have selected '{0}' is invalid")]
    InvalidMidiPortSelected(String),
    #[error("Pixels encountered an issue: {0}")]
    Pixels(#[from] pixels::Error),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Config(#[from] config::ConfigError),
    #[error("{0}")]
    ThreadSafety(String),
}
