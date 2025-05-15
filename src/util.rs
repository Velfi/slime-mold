use num::NumCast;

// Stolen from [nannou](https://docs.rs/nannou/0.15.0/src/nannou/math.rs.html#42)
pub fn map_range<X, Y>(val: X, in_min: X, in_max: X, out_min: Y, out_max: Y) -> Y
where
    X: NumCast,
    Y: NumCast,
{
    let val_f: f32 = NumCast::from(val)
        .unwrap_or_else(|| panic!("[map_range] failed to cast first arg to `f32`"));
    let in_min_f: f32 = NumCast::from(in_min)
        .unwrap_or_else(|| panic!("[map_range] failed to cast second arg to `f32`"));
    let in_max_f: f32 = NumCast::from(in_max)
        .unwrap_or_else(|| panic!("[map_range] failed to cast third arg to `f32`"));
    let out_min_f: f32 = NumCast::from(out_min)
        .unwrap_or_else(|| panic!("[map_range] failed to cast fourth arg to `f32`"));
    let out_max_f: f32 = NumCast::from(out_max)
        .unwrap_or_else(|| panic!("[map_range] failed to cast fifth arg to `f32`"));

    NumCast::from((val_f - in_min_f) / (in_max_f - in_min_f) * (out_max_f - out_min_f) + out_min_f)
        .unwrap_or_else(|| panic!("[map_range] failed to cast result to target type"))
}
