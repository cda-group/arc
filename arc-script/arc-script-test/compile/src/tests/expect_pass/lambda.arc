# Closures are not yet supported by arc-script
# XFAIL: *
# RUN: arc-mlir-rust-test %t %s

fun test(): i32 {
    val increment = fun(i): i + 1;
    val foo = 1;
    increment(foo)
}
