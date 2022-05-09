// XFAIL: *
// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustfloattensorarith {
func.func @addf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.addf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func.func @subf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.subf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func.func @mulf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.mulf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func.func @divf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.divf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func.func @remf_tensor2x2xf32(%a : tensor<2x2xf32>, %b : tensor<2x2xf32>) -> tensor<2x2xf32> {
  %c = arith.remf %a, %b : tensor<2x2xf32>
  return %c : tensor<2x2xf32>
}

func.func @addf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.addf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func.func @subf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.subf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func.func @mulf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.mulf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func.func @divf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.divf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

func.func @remf_tensor2x2xf64(%a : tensor<2x2xf64>, %b : tensor<2x2xf64>) -> tensor<2x2xf64> {
  %c = arith.remf %a, %b : tensor<2x2xf64>
  return %c : tensor<2x2xf64>
}

}
