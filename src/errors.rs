use midir::MidiInput;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SlimeError {
    #[error("Midir encountered an issue: {0}")]
    MidirConnection(#[from] midir::ConnectError<MidiInput>),
    #[error("Couldn't detect any MIDI devices")]
    NoMidiDevicesDetected,
    #[error("Couldn't find a MIDI device with the name '{0}', please check your spelling")]
    NoMidiDevicesWithName(String),
    #[error("The MIDI port you have selected '{0}' is invalid")]
    InvalidMidiPortSelected(String),
    #[error("Pixels encountered an issue: {0}")]
    Pixels(#[from] pixels::Error),
}
