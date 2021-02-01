# RUN: arc-script --emit-mlir check %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/unknown/Cargo.toml

fun fib(n: i32) -> i32 {
    if n > 2 {
        fib(n - 1) + fib(n - 2)
    } else {
        0
    }
}

fun test() -> i32 {
  fib(5)
}
