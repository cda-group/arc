// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && CARGO_HTTP_DEBUG=true cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
module @toplevel {
func @addi_tensor2x2xui8(%a : tensor<2x2xui8>, %b : tensor<2x2xui8>) -> tensor<2x2xui8> {
  %c = arc.addi %a, %b : tensor<2x2xui8>
  return %c : tensor<2x2xui8>
}

func @subi_tensor2x2xui8(%a : tensor<2x2xui8>, %b : tensor<2x2xui8>) -> tensor<2x2xui8> {
  %c = arc.subi %a, %b : tensor<2x2xui8>
  return %c : tensor<2x2xui8>
}

func @muli_tensor2x2xui8(%a : tensor<2x2xui8>, %b : tensor<2x2xui8>) -> tensor<2x2xui8> {
  %c = arc.muli %a, %b : tensor<2x2xui8>
  return %c : tensor<2x2xui8>
}

func @divi_tensor2x2xui8(%a : tensor<2x2xui8>, %b : tensor<2x2xui8>) -> tensor<2x2xui8> {
  %c = arc.divi %a, %b : tensor<2x2xui8>
  return %c : tensor<2x2xui8>
}

func @remi_tensor2x2xui8(%a : tensor<2x2xui8>, %b : tensor<2x2xui8>) -> tensor<2x2xui8> {
  %c = arc.remi %a, %b : tensor<2x2xui8>
  return %c : tensor<2x2xui8>
}

func @addi_tensor2x2xui16(%a : tensor<2x2xui16>, %b : tensor<2x2xui16>) -> tensor<2x2xui16> {
  %c = arc.addi %a, %b : tensor<2x2xui16>
  return %c : tensor<2x2xui16>
}

func @subi_tensor2x2xui16(%a : tensor<2x2xui16>, %b : tensor<2x2xui16>) -> tensor<2x2xui16> {
  %c = arc.subi %a, %b : tensor<2x2xui16>
  return %c : tensor<2x2xui16>
}

func @muli_tensor2x2xui16(%a : tensor<2x2xui16>, %b : tensor<2x2xui16>) -> tensor<2x2xui16> {
  %c = arc.muli %a, %b : tensor<2x2xui16>
  return %c : tensor<2x2xui16>
}

func @divi_tensor2x2xui16(%a : tensor<2x2xui16>, %b : tensor<2x2xui16>) -> tensor<2x2xui16> {
  %c = arc.divi %a, %b : tensor<2x2xui16>
  return %c : tensor<2x2xui16>
}

func @remi_tensor2x2xui16(%a : tensor<2x2xui16>, %b : tensor<2x2xui16>) -> tensor<2x2xui16> {
  %c = arc.remi %a, %b : tensor<2x2xui16>
  return %c : tensor<2x2xui16>
}

func @addi_tensor2x2xui32(%a : tensor<2x2xui32>, %b : tensor<2x2xui32>) -> tensor<2x2xui32> {
  %c = arc.addi %a, %b : tensor<2x2xui32>
  return %c : tensor<2x2xui32>
}

func @subi_tensor2x2xui32(%a : tensor<2x2xui32>, %b : tensor<2x2xui32>) -> tensor<2x2xui32> {
  %c = arc.subi %a, %b : tensor<2x2xui32>
  return %c : tensor<2x2xui32>
}

func @muli_tensor2x2xui32(%a : tensor<2x2xui32>, %b : tensor<2x2xui32>) -> tensor<2x2xui32> {
  %c = arc.muli %a, %b : tensor<2x2xui32>
  return %c : tensor<2x2xui32>
}

func @divi_tensor2x2xui32(%a : tensor<2x2xui32>, %b : tensor<2x2xui32>) -> tensor<2x2xui32> {
  %c = arc.divi %a, %b : tensor<2x2xui32>
  return %c : tensor<2x2xui32>
}

func @remi_tensor2x2xui32(%a : tensor<2x2xui32>, %b : tensor<2x2xui32>) -> tensor<2x2xui32> {
  %c = arc.remi %a, %b : tensor<2x2xui32>
  return %c : tensor<2x2xui32>
}

func @addi_tensor2x2xui64(%a : tensor<2x2xui64>, %b : tensor<2x2xui64>) -> tensor<2x2xui64> {
  %c = arc.addi %a, %b : tensor<2x2xui64>
  return %c : tensor<2x2xui64>
}

func @subi_tensor2x2xui64(%a : tensor<2x2xui64>, %b : tensor<2x2xui64>) -> tensor<2x2xui64> {
  %c = arc.subi %a, %b : tensor<2x2xui64>
  return %c : tensor<2x2xui64>
}

func @muli_tensor2x2xui64(%a : tensor<2x2xui64>, %b : tensor<2x2xui64>) -> tensor<2x2xui64> {
  %c = arc.muli %a, %b : tensor<2x2xui64>
  return %c : tensor<2x2xui64>
}

func @divi_tensor2x2xui64(%a : tensor<2x2xui64>, %b : tensor<2x2xui64>) -> tensor<2x2xui64> {
  %c = arc.divi %a, %b : tensor<2x2xui64>
  return %c : tensor<2x2xui64>
}

func @remi_tensor2x2xui64(%a : tensor<2x2xui64>, %b : tensor<2x2xui64>) -> tensor<2x2xui64> {
  %c = arc.remi %a, %b : tensor<2x2xui64>
  return %c : tensor<2x2xui64>
}

func @addi_tensor2x2xsi8(%a : tensor<2x2xsi8>, %b : tensor<2x2xsi8>) -> tensor<2x2xsi8> {
  %c = arc.addi %a, %b : tensor<2x2xsi8>
  return %c : tensor<2x2xsi8>
}

func @subi_tensor2x2xsi8(%a : tensor<2x2xsi8>, %b : tensor<2x2xsi8>) -> tensor<2x2xsi8> {
  %c = arc.subi %a, %b : tensor<2x2xsi8>
  return %c : tensor<2x2xsi8>
}

func @muli_tensor2x2xsi8(%a : tensor<2x2xsi8>, %b : tensor<2x2xsi8>) -> tensor<2x2xsi8> {
  %c = arc.muli %a, %b : tensor<2x2xsi8>
  return %c : tensor<2x2xsi8>
}

func @divi_tensor2x2xsi8(%a : tensor<2x2xsi8>, %b : tensor<2x2xsi8>) -> tensor<2x2xsi8> {
  %c = arc.divi %a, %b : tensor<2x2xsi8>
  return %c : tensor<2x2xsi8>
}

func @remi_tensor2x2xsi8(%a : tensor<2x2xsi8>, %b : tensor<2x2xsi8>) -> tensor<2x2xsi8> {
  %c = arc.remi %a, %b : tensor<2x2xsi8>
  return %c : tensor<2x2xsi8>
}

func @addi_tensor2x2xsi16(%a : tensor<2x2xsi16>, %b : tensor<2x2xsi16>) -> tensor<2x2xsi16> {
  %c = arc.addi %a, %b : tensor<2x2xsi16>
  return %c : tensor<2x2xsi16>
}

func @subi_tensor2x2xsi16(%a : tensor<2x2xsi16>, %b : tensor<2x2xsi16>) -> tensor<2x2xsi16> {
  %c = arc.subi %a, %b : tensor<2x2xsi16>
  return %c : tensor<2x2xsi16>
}

func @muli_tensor2x2xsi16(%a : tensor<2x2xsi16>, %b : tensor<2x2xsi16>) -> tensor<2x2xsi16> {
  %c = arc.muli %a, %b : tensor<2x2xsi16>
  return %c : tensor<2x2xsi16>
}

func @divi_tensor2x2xsi16(%a : tensor<2x2xsi16>, %b : tensor<2x2xsi16>) -> tensor<2x2xsi16> {
  %c = arc.divi %a, %b : tensor<2x2xsi16>
  return %c : tensor<2x2xsi16>
}

func @remi_tensor2x2xsi16(%a : tensor<2x2xsi16>, %b : tensor<2x2xsi16>) -> tensor<2x2xsi16> {
  %c = arc.remi %a, %b : tensor<2x2xsi16>
  return %c : tensor<2x2xsi16>
}

func @addi_tensor2x2xsi32(%a : tensor<2x2xsi32>, %b : tensor<2x2xsi32>) -> tensor<2x2xsi32> {
  %c = arc.addi %a, %b : tensor<2x2xsi32>
  return %c : tensor<2x2xsi32>
}

func @subi_tensor2x2xsi32(%a : tensor<2x2xsi32>, %b : tensor<2x2xsi32>) -> tensor<2x2xsi32> {
  %c = arc.subi %a, %b : tensor<2x2xsi32>
  return %c : tensor<2x2xsi32>
}

func @muli_tensor2x2xsi32(%a : tensor<2x2xsi32>, %b : tensor<2x2xsi32>) -> tensor<2x2xsi32> {
  %c = arc.muli %a, %b : tensor<2x2xsi32>
  return %c : tensor<2x2xsi32>
}

func @divi_tensor2x2xsi32(%a : tensor<2x2xsi32>, %b : tensor<2x2xsi32>) -> tensor<2x2xsi32> {
  %c = arc.divi %a, %b : tensor<2x2xsi32>
  return %c : tensor<2x2xsi32>
}

func @remi_tensor2x2xsi32(%a : tensor<2x2xsi32>, %b : tensor<2x2xsi32>) -> tensor<2x2xsi32> {
  %c = arc.remi %a, %b : tensor<2x2xsi32>
  return %c : tensor<2x2xsi32>
}

func @addi_tensor2x2xsi64(%a : tensor<2x2xsi64>, %b : tensor<2x2xsi64>) -> tensor<2x2xsi64> {
  %c = arc.addi %a, %b : tensor<2x2xsi64>
  return %c : tensor<2x2xsi64>
}

func @subi_tensor2x2xsi64(%a : tensor<2x2xsi64>, %b : tensor<2x2xsi64>) -> tensor<2x2xsi64> {
  %c = arc.subi %a, %b : tensor<2x2xsi64>
  return %c : tensor<2x2xsi64>
}

func @muli_tensor2x2xsi64(%a : tensor<2x2xsi64>, %b : tensor<2x2xsi64>) -> tensor<2x2xsi64> {
  %c = arc.muli %a, %b : tensor<2x2xsi64>
  return %c : tensor<2x2xsi64>
}

func @divi_tensor2x2xsi64(%a : tensor<2x2xsi64>, %b : tensor<2x2xsi64>) -> tensor<2x2xsi64> {
  %c = arc.divi %a, %b : tensor<2x2xsi64>
  return %c : tensor<2x2xsi64>
}

func @remi_tensor2x2xsi64(%a : tensor<2x2xsi64>, %b : tensor<2x2xsi64>) -> tensor<2x2xsi64> {
  %c = arc.remi %a, %b : tensor<2x2xsi64>
  return %c : tensor<2x2xsi64>
}

}
