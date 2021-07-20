# RUN: arc-script run --output=MLIR %s | arc-mlir

task Identity(): (A(~i32 by i32)) -> (B(~i32 by i32)) {
    on A(event) => emit B(event);
}

fun main(input: ~i32 by i32): ~i32 by i32 {
    val output = input | Identity();
    output
}
