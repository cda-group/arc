# RUN: arc-script run --output=MLIR %s | arc-mlir -rustcratename expectpassiflet -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/expectpassiflet/Cargo.toml
fun test() -> i32 {
    let x = 3 in
    let y = 5 in
    if let ((1, z), 1) = ((3, x), 1) {
        z+y
    } else {
        y+2
    }
}
