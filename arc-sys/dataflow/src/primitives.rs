pub fn assert(b: bool) {
    assert!(b);
}

pub fn panic(s: Str) {
    panic!("{}", s.as_str())
}

pub fn print(s: Str) {
    tracing::info!("{}", s.as_str())
}
