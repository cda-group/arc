# RUN: arc-script run --output=MLIR %s | arc-mlir

fun main() {
    let ((a, b), (c, d)) = ((1, 2), (3, 4)) in
    ()
}
