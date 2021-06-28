# RUN: arc-script run --output=MLIR %s | arc-mlir

enum Opt {
    Some(i32),
    None,
}

fun main() {
    if val Opt::Some(x) = Opt::Some(5) {
        unit
    }
}
