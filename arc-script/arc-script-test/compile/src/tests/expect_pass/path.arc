# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir

fun main() {
    val x = 1;
    val y = x;
}
