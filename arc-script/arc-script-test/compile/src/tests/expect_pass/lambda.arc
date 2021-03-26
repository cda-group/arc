# Closures are not yet supported by arc-script
# XFAIL: *
# RUN: arc-script --emit-mlir check %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/unknown/Cargo.toml

fun test(): i32 {
    let increment = fun(i): i + 1 in
    let foo = 1 in
    increment(foo)
}
