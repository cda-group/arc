// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustsimple {
  func.func @returnf64() -> f64 {
    %b = arith.constant 3.14 : f64
    return %b : f64
  }
  func.func @returnf32() -> f32 {
    %b = arith.constant 0.69315 : f32
    return %b : f32
  }
  func.func @return_true() -> i1 {
    %b = arith.constant 1 : i1
    return %b : i1
  }
  func.func @return_false() -> i1 {
    %b = arith.constant 0 : i1
    return %b : i1
  }
  func.func @return_ui8() -> ui8 {
    %b = arc.constant 255 : ui8
    return %b : ui8
  }
  func.func @return_ui16() -> ui16 {
    %b = arc.constant 65535 : ui16
    return %b : ui16
  }
  func.func @return_ui32() -> ui32 {
    %b = arc.constant 4294967295 : ui32
    return %b : ui32
  }
  func.func @return_ui64() -> ui64 {
    %b = arc.constant 18446744073709551615 : ui64
    return %b : ui64
  }
  func.func @return_si8() -> si8 {
    %b = arc.constant -128 : si8
    return %b : si8
  }
  func.func @return_si16() -> si16 {
    %b = arc.constant -32768 : si16
    return %b : si16
  }
  func.func @return_si32() -> si32 {
    %b = arc.constant -2147483648 : si32
    return %b : si32
  }
  func.func @return_si64() -> si64 {
    %b = arc.constant -9223372036854775808 : si64
    return %b : si64
  }
}
