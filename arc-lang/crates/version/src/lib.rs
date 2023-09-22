mod data {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}

pub fn show() {
    println!(" version: {}", data::PKG_VERSION);
    println!("  target: {}", data::TARGET);
    println!(" profile: {}", data::PROFILE);
    println!("compiler: {}", data::RUSTC_VERSION);
    println!("    time: {}", data::BUILT_TIME_UTC);
}
