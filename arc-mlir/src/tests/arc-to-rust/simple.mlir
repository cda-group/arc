// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {
  func @returnf64() -> f64 {
    %b = constant 3.14 : f64
    return %b : f64
  }
  func @returnf32() -> f32 {
    %b = constant 0.69315 : f32
    return %b : f32
  }
  func @return_true() -> i1 {
    %b = constant 1 : i1
    return %b : i1
  }
  func @return_false() -> i1 {
    %b = constant 0 : i1
    return %b : i1
  }
}
