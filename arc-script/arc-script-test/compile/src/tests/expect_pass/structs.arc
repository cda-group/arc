# RUN: arc-script run --output=MLIR %s | arc-mlir -arc-to-rust

fun foo(a: {c: i32, b: i32}, b: {b:i32, c:i32}): {a:{c:i32, b:i32}, xyz:i32, d:{b:i32, c:i32}} {
    let r = {a:a, xyz:4711, d:b} in
    r
}

fun test() {
    let x = foo({c:1, b:5}, {b:4, c:2}) in
    unit
}
