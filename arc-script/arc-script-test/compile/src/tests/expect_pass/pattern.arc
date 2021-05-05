# RUN: arc-script run --output=MLIR %s | arc-mlir

fun main() {
    val ((a, b), (c, d)) = ((1, 2), (3, 4));
}
