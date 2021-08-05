# Closures are not yet supported by arc-script
# XFAIL: *
# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/unknown/Cargo.toml

fun test(): i32 {
    val increment = fun(i): i + 1;
    val foo = 1;
    increment(foo)
}
