// RUN: arc-mlir -arc-to-rust -crate %t %s && cargo build -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t %s && cargo build -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {
  func @main() -> f64 {
    %b = constant 3.14 : f64
    return %b : f64
  }
}
