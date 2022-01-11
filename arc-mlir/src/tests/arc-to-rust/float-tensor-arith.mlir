// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustfloattensorarith {
func @addf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.addf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @subf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.subf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @mulf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.mulf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @divf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.divf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @remf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.remf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func @addf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.addf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func @subf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.subf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func @mulf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.mulf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func @divf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.divf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func @remf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.remf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

}
