# RUN: arc-script run --output=MLIR %s | arc-mlir -rustcratename expectpassbasic -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/expectpassbasic/Cargo.toml

fun main() {
  unit
}

fun test() -> i32 {
  1
}
