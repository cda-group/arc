# RUN: arc-script run --output=MLIR %s | arc-mlir

fun foo(x: unit): unit {
    x
}

fun bar() {
    foo(unit)
}
