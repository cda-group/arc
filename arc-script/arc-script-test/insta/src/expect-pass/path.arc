# RUN: arc-script run --output=MLIR %s | arc-mlir

fun main() {
  let x = 1 in
  let y = x in
  unit
}
