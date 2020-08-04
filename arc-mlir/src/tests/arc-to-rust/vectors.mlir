// RUN: arc-mlir -arc-to-rust -crate %t %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {
  func @in_out_0(%0 : tensor<4xsi32>) -> tensor<4xsi32> {
    return %0 : tensor<4xsi32>
  }
  func @in_out_1(%0 : tensor<4x5xsi32>) -> tensor<4x5xsi32> {
    return %0 : tensor<4x5xsi32>
  }
  func @in_out_2(%0 : tensor<?xsi32>) -> tensor<?xsi32> {
    return %0 : tensor<?xsi32>
  }

}
