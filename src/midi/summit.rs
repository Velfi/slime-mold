/// Controller codes for my Novation Summit
/// For more info, see page 47 of the [Summit User Manual](https://fael-downloads-prod.focusrite.com/customer/prod/s3fs-public/downloads/Summit%20manual%201.0.1.pdf)
pub enum SummitController {
    // Filter frequency ranges from 0-255 so it's represented with a pair of u7s
    NoiseVolumeA = 27,
    NoiseVolumeB = 59,
    // Filter frequency ranges from 0-255 so it's represented with a pair of u7s
    // Not sure how to handle u7 pairs yet
    FilterFrequencyA = 29,
    FilterFrequencyB = 61,
    Other,
}

impl From<u8> for SummitController {
    fn from(n: u8) -> Self {
        use SummitController::*;
        match n {
            27 => NoiseVolumeA,
            59 => NoiseVolumeB,
            29 => FilterFrequencyA,
            61 => FilterFrequencyB,
            _ => Other,
        }
    }
}
