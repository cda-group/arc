fn main() {
    if std::env::var("IGNORE_WORKER_BUILD_SCRIPT").is_ok() {
        return;
    }
    println!(
        "cargo:rustc-env=TARGET={}",
        std::env::var("TARGET").unwrap()
    );
}
