# XFAIL: *
# RUN: arc-script run --output=MLIR %s | arc-mlir

enum Option {
    Some(i32),
    None
}

fun main() {
    if let Option::Some(y) = Option::Some(3) {
        unit
    } else {
        unit
    }
}
