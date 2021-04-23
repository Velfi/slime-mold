use crate::errors::SlimeError;
use log::{error, info, trace, warn};
use midir::{Ignore, MidiInput, MidiInputConnection, MidiInputPort};
use midly::{live::LiveEvent, MidiMessage};
use std::io::{stdin, stdout, Write};
use std::sync::mpsc;
use std::{mem, sync::mpsc::Receiver};

pub struct MidiInterface {
    connection: MidiConnection,
    in_port: MidiInputPort,
    event_receiver: Option<Receiver<MidiMessage>>,
}

enum MidiConnection {
    Unconnected(MidiInput),
    Pending,
    Connected(MidiInputConnection<()>),
}

impl MidiInterface {
    pub fn new(port_name: Option<&str>) -> Result<Self, SlimeError> {
        let mut midi_in = MidiInput::new("midir").expect("creating midi client");
        midi_in.ignore(Ignore::None);

        let in_ports = midi_in.ports();
        let in_port = match in_ports.len() {
            0 => Err(SlimeError::NoMidiDevicesDetected),
            1 => {
                info!(
                    "Automatically choosing the only available MIDI input port: {}",
                    midi_in.port_name(&in_ports[0]).unwrap()
                );
                Ok(in_ports[0].clone())
            }
            _ => match port_name {
                Some(port_name) => {
                    info!("opening connection to MIDI port {}", port_name);
                    let mut port_result =
                        Err(SlimeError::NoMidiDevicesWithName(port_name.to_owned()));

                    for port in in_ports.iter() {
                        if midi_in.port_name(port).unwrap() == port_name {
                            port_result = Ok(port.clone());
                            break;
                        }
                    }

                    port_result
                }
                None => {
                    info!("Available input ports:");
                    for (i, p) in in_ports.iter().enumerate() {
                        info!("{}: {}", i, midi_in.port_name(p).unwrap());
                    }
                    info!("Please select input port: ");
                    stdout().flush().unwrap();
                    let mut input = String::new();
                    stdin().read_line(&mut input).unwrap();
                    in_ports
                        .get(input.trim().parse::<usize>().unwrap())
                        .cloned()
                        .ok_or_else(|| SlimeError::InvalidMidiPortSelected(input))
                }
            },
        }?;

        Ok(Self {
            connection: MidiConnection::Unconnected(midi_in),
            in_port,
            event_receiver: None,
        })
    }

    pub fn open(&mut self) -> Result<(), SlimeError> {
        info!("\nOpening MIDI connection");
        let (sender, receiver) = mpsc::channel();
        self.event_receiver = Some(receiver);
        let port = self.in_port.clone();
        if let MidiConnection::Unconnected(midi_in) =
            mem::replace(&mut self.connection, MidiConnection::Pending)
        {
            let port_name = midi_in
                .port_name(&self.in_port)
                .expect("couldn't get midi port name");

            // _conn_in needs to be a named parameter, because it needs to be kept alive until the end of the scope
            let conn_in = midi_in.connect(
                &port,
                "midir-read-input",
                move |_stamp, message, _| {
                    //   match message {
                    //     &[a, b, c] => {
                    //       if let Err(err) = sender.send(MidiEvent { stamp, message: [a, b, c] }) {
                    //         error!("Couldn't send MIDI message through MPSC with error: {}", err);
                    //       }
                    //     },
                    //     _ => {
                    //       error!("received a MIDI message {} bytes long but expected a 3 byte long message", message.len());
                    //     }
                    //   };
                    let message = message.to_owned();
                    match LiveEvent::parse(&message) {
                        Ok(LiveEvent::Midi { message, .. }) => {
                            if let Err(err) = sender.send(message) {
                                error!(
                                    "Couldn't send MIDI message through MPSC with error: {}",
                                    err
                                );
                            }
                        }
                        Ok(other_event) => {
                            warn!("Received an unhandled MIDI event: {:#?}", other_event)
                        }
                        Err(err) => {
                            error!("Couldn't parse MIDI message: {}", err)
                        }
                    }
                },
                (),
            )?;

            self.connection = MidiConnection::Connected(conn_in);

            info!(
                "MIDI connection open, reading input from port '{}'",
                port_name
            );

            Ok(())
        } else {
            unreachable!("it's up to the consumer to never call this more than once")
        }
    }

    pub fn pending_events(&self) -> Box<dyn Iterator<Item = MidiMessage>> {
        let iter: Box<dyn Iterator<Item = MidiMessage>> = if let Some(rcv) = &self.event_receiver {
            trace!("MIDI receiver is opened, pending events may exist");
            // I couldn't figure out any other way to do this
            let i: Vec<_> = rcv.try_iter().collect();
            Box::new(i.into_iter())
        } else {
            warn!("No MIDI event receiver has been opened. Please open the MidiInterface before checking for pending events");
            Box::new(std::iter::empty())
        };

        iter
    }
}
