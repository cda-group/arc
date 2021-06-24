# RUN: arc-script run --output=MLIR %s | arc-mlir

task Map(f: fun(i32): i32): ~i32 by i32 -> ~i32 by i32 {
    on event by key => emit f(event) by key;
}

fun main(input: ~i32 by i32): ~i32 by i32 {
    val output = input | Map(fun(x): x + 1);
    output
}
