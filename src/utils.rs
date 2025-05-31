pub fn format_float_dynamic(val: f32) -> String {
    let s = format!("{}", val);
    if s.contains('.') {
        let s = s.trim_end_matches('0').trim_end_matches('.');
        if s.is_empty() {
            "0.0".to_string()
        } else {
            s.to_string()
        }
    } else {
        format!("{}.0", s)
    }
} 