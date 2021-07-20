# RUN: arc-script run --output=MLIR %s | arc-mlir

task Identity(): ~i32 by i32 -> ~i32 by i32 {
    on event => emit event;
}
