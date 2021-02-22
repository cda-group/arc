# RUN: arc-script run --output=MLIR %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

fun test() {
    let x = {a:{c:1, b:5}, xyz:2, d:{b:4, c:2}} in
    unit
}
