// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustintbitops {
  func.func @and_ui8(%arg0: ui8, %arg1: ui8) -> ui8 {
    %0 = arc.and %arg0, %arg1 : ui8
    return %0 : ui8
  }
  func.func @or_ui8(%arg0: ui8, %arg1: ui8) -> ui8 {
    %0 = arc.or %arg0, %arg1 : ui8
    return %0 : ui8
  }
  func.func @xor_ui8(%arg0: ui8, %arg1: ui8) -> ui8 {
    %0 = arc.xor %arg0, %arg1 : ui8
    return %0 : ui8
  }
  func.func @and_ui16(%arg0: ui16, %arg1: ui16) -> ui16 {
    %0 = arc.and %arg0, %arg1 : ui16
    return %0 : ui16
  }
  func.func @or_ui16(%arg0: ui16, %arg1: ui16) -> ui16 {
    %0 = arc.or %arg0, %arg1 : ui16
    return %0 : ui16
  }
  func.func @xor_ui16(%arg0: ui16, %arg1: ui16) -> ui16 {
    %0 = arc.xor %arg0, %arg1 : ui16
    return %0 : ui16
  }
  func.func @and_ui32(%arg0: ui32, %arg1: ui32) -> ui32 {
    %0 = arc.and %arg0, %arg1 : ui32
    return %0 : ui32
  }
  func.func @or_ui32(%arg0: ui32, %arg1: ui32) -> ui32 {
    %0 = arc.or %arg0, %arg1 : ui32
    return %0 : ui32
  }
  func.func @xor_ui32(%arg0: ui32, %arg1: ui32) -> ui32 {
    %0 = arc.xor %arg0, %arg1 : ui32
    return %0 : ui32
  }
  func.func @and_ui64(%arg0: ui64, %arg1: ui64) -> ui64 {
    %0 = arc.and %arg0, %arg1 : ui64
    return %0 : ui64
  }
  func.func @or_ui64(%arg0: ui64, %arg1: ui64) -> ui64 {
    %0 = arc.or %arg0, %arg1 : ui64
    return %0 : ui64
  }
  func.func @xor_ui64(%arg0: ui64, %arg1: ui64) -> ui64 {
    %0 = arc.xor %arg0, %arg1 : ui64
    return %0 : ui64
  }
  func.func @and_si8(%arg0: si8, %arg1: si8) -> si8 {
    %0 = arc.and %arg0, %arg1 : si8
    return %0 : si8
  }
  func.func @or_si8(%arg0: si8, %arg1: si8) -> si8 {
    %0 = arc.or %arg0, %arg1 : si8
    return %0 : si8
  }
  func.func @xor_si8(%arg0: si8, %arg1: si8) -> si8 {
    %0 = arc.xor %arg0, %arg1 : si8
    return %0 : si8
  }
  func.func @and_si16(%arg0: si16, %arg1: si16) -> si16 {
    %0 = arc.and %arg0, %arg1 : si16
    return %0 : si16
  }
  func.func @or_si16(%arg0: si16, %arg1: si16) -> si16 {
    %0 = arc.or %arg0, %arg1 : si16
    return %0 : si16
  }
  func.func @xor_si16(%arg0: si16, %arg1: si16) -> si16 {
    %0 = arc.xor %arg0, %arg1 : si16
    return %0 : si16
  }
  func.func @and_si32(%arg0: si32, %arg1: si32) -> si32 {
    %0 = arc.and %arg0, %arg1 : si32
    return %0 : si32
  }
  func.func @or_si32(%arg0: si32, %arg1: si32) -> si32 {
    %0 = arc.or %arg0, %arg1 : si32
    return %0 : si32
  }
  func.func @xor_si32(%arg0: si32, %arg1: si32) -> si32 {
    %0 = arc.xor %arg0, %arg1 : si32
    return %0 : si32
  }
  func.func @and_si64(%arg0: si64, %arg1: si64) -> si64 {
    %0 = arc.and %arg0, %arg1 : si64
    return %0 : si64
  }
  func.func @or_si64(%arg0: si64, %arg1: si64) -> si64 {
    %0 = arc.or %arg0, %arg1 : si64
    return %0 : si64
  }
  func.func @xor_si64(%arg0: si64, %arg1: si64) -> si64 {
    %0 = arc.xor %arg0, %arg1 : si64
    return %0 : si64
  }
}
