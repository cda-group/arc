// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
module @toplevel {
func @addf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = std.addf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @subf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = std.subf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @mulf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = std.mulf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @divf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = std.divf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @remf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = std.remf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @addf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = std.addf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func @subf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = std.subf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func @mulf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = std.mulf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func @divf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = std.divf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func @remf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = std.remf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

}
