# RUN: arc-script run --output=MLIR %s | arc-mlir -rustcratename expectrunfun -arc-to-rust

fun max(a: i32, b: i32): i32 {
    val c = a > b;
    if c {
        a+1
    } else {
        b
    }
}

fun test(): i32 {
    max(1, 2)
}
