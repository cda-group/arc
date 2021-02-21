# RUN: arc-script run --output=MLIR %s | arc-mlir -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

fun main() {
  unit
}

fun test() -> i32 {
  1
}
