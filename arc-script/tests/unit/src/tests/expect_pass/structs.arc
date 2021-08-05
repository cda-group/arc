# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir -arc-to-rust

fun foo(a: {c: i32, b: i32}, b: {b:i32, c:i32}): {a:{c:i32, b:i32}, xyz:i32, d:{b:i32, c:i32}} {
    val r = {a:a, xyz:4711, d:b};
    r
}

fun test() {
    val x = foo({c:1, b:5}, {b:4, c:2});
}
