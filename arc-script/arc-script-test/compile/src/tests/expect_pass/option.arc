# XFAIL: *
# RUN: arc-script run --output=MLIR %s | arc-mlir

enum Opt {
    Some(i32),
    None
}

fun main() {
    if val Opt::Some(y) = Opt::Some(3) {
        unit
    }
}
