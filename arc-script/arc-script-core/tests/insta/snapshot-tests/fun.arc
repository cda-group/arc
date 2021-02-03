# RUN: arc-script run --output=MLIR %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

fun max(a: i32, b: i32) -> i32 {
    let c = a > b in
    if c {
        a
    } else {
        b
    }
}

fun test() -> i32 {
    max(1, 2)
}
