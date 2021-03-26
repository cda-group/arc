# RUN: arc-script run --output=MLIR %s | arc-mlir -rustcratename expectpassfib -arc-to-rust

fun fib(n: i32): i32 {
    if n > 2 {
        fib(n - 1) + fib(n - 2)
    } else {
        0
    }
}

fun test(): i32 {
  fib(5)
}
