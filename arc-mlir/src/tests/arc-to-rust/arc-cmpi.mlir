// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustcmpi {
  func.func @eq_ui8(%a : ui8, %b : ui8) -> i1 {
    %r = arc.cmpi "eq", %a, %b : ui8
    return %r : i1
  }

  func.func @ne_ui8(%a : ui8, %b : ui8) -> i1 {
    %r = arc.cmpi "ne", %a, %b : ui8
    return %r : i1
  }

  func.func @lt_ui8(%a : ui8, %b : ui8) -> i1 {
    %r = arc.cmpi "lt", %a, %b : ui8
    return %r : i1
  }

  func.func @le_ui8(%a : ui8, %b : ui8) -> i1 {
    %r = arc.cmpi "le", %a, %b : ui8
    return %r : i1
  }

  func.func @gt_ui8(%a : ui8, %b : ui8) -> i1 {
    %r = arc.cmpi "gt", %a, %b : ui8
    return %r : i1
  }

  func.func @ge_ui8(%a : ui8, %b : ui8) -> i1 {
    %r = arc.cmpi "ge", %a, %b : ui8
    return %r : i1
  }

  func.func @eq_ui16(%a : ui16, %b : ui16) -> i1 {
    %r = arc.cmpi "eq", %a, %b : ui16
    return %r : i1
  }

  func.func @ne_ui16(%a : ui16, %b : ui16) -> i1 {
    %r = arc.cmpi "ne", %a, %b : ui16
    return %r : i1
  }

  func.func @lt_ui16(%a : ui16, %b : ui16) -> i1 {
    %r = arc.cmpi "lt", %a, %b : ui16
    return %r : i1
  }

  func.func @le_ui16(%a : ui16, %b : ui16) -> i1 {
    %r = arc.cmpi "le", %a, %b : ui16
    return %r : i1
  }

  func.func @gt_ui16(%a : ui16, %b : ui16) -> i1 {
    %r = arc.cmpi "gt", %a, %b : ui16
    return %r : i1
  }

  func.func @ge_ui16(%a : ui16, %b : ui16) -> i1 {
    %r = arc.cmpi "ge", %a, %b : ui16
    return %r : i1
  }

  func.func @eq_ui32(%a : ui32, %b : ui32) -> i1 {
    %r = arc.cmpi "eq", %a, %b : ui32
    return %r : i1
  }

  func.func @ne_ui32(%a : ui32, %b : ui32) -> i1 {
    %r = arc.cmpi "ne", %a, %b : ui32
    return %r : i1
  }

  func.func @lt_ui32(%a : ui32, %b : ui32) -> i1 {
    %r = arc.cmpi "lt", %a, %b : ui32
    return %r : i1
  }

  func.func @le_ui32(%a : ui32, %b : ui32) -> i1 {
    %r = arc.cmpi "le", %a, %b : ui32
    return %r : i1
  }

  func.func @gt_ui32(%a : ui32, %b : ui32) -> i1 {
    %r = arc.cmpi "gt", %a, %b : ui32
    return %r : i1
  }

  func.func @ge_ui32(%a : ui32, %b : ui32) -> i1 {
    %r = arc.cmpi "ge", %a, %b : ui32
    return %r : i1
  }

  func.func @eq_ui64(%a : ui64, %b : ui64) -> i1 {
    %r = arc.cmpi "eq", %a, %b : ui64
    return %r : i1
  }

  func.func @ne_ui64(%a : ui64, %b : ui64) -> i1 {
    %r = arc.cmpi "ne", %a, %b : ui64
    return %r : i1
  }

  func.func @lt_ui64(%a : ui64, %b : ui64) -> i1 {
    %r = arc.cmpi "lt", %a, %b : ui64
    return %r : i1
  }

  func.func @le_ui64(%a : ui64, %b : ui64) -> i1 {
    %r = arc.cmpi "le", %a, %b : ui64
    return %r : i1
  }

  func.func @gt_ui64(%a : ui64, %b : ui64) -> i1 {
    %r = arc.cmpi "gt", %a, %b : ui64
    return %r : i1
  }

  func.func @ge_ui64(%a : ui64, %b : ui64) -> i1 {
    %r = arc.cmpi "ge", %a, %b : ui64
    return %r : i1
  }

  func.func @eq_si8(%a : si8, %b : si8) -> i1 {
    %r = arc.cmpi "eq", %a, %b : si8
    return %r : i1
  }

  func.func @ne_si8(%a : si8, %b : si8) -> i1 {
    %r = arc.cmpi "ne", %a, %b : si8
    return %r : i1
  }

  func.func @lt_si8(%a : si8, %b : si8) -> i1 {
    %r = arc.cmpi "lt", %a, %b : si8
    return %r : i1
  }

  func.func @le_si8(%a : si8, %b : si8) -> i1 {
    %r = arc.cmpi "le", %a, %b : si8
    return %r : i1
  }

  func.func @gt_si8(%a : si8, %b : si8) -> i1 {
    %r = arc.cmpi "gt", %a, %b : si8
    return %r : i1
  }

  func.func @ge_si8(%a : si8, %b : si8) -> i1 {
    %r = arc.cmpi "ge", %a, %b : si8
    return %r : i1
  }

  func.func @eq_si16(%a : si16, %b : si16) -> i1 {
    %r = arc.cmpi "eq", %a, %b : si16
    return %r : i1
  }

  func.func @ne_si16(%a : si16, %b : si16) -> i1 {
    %r = arc.cmpi "ne", %a, %b : si16
    return %r : i1
  }

  func.func @lt_si16(%a : si16, %b : si16) -> i1 {
    %r = arc.cmpi "lt", %a, %b : si16
    return %r : i1
  }

  func.func @le_si16(%a : si16, %b : si16) -> i1 {
    %r = arc.cmpi "le", %a, %b : si16
    return %r : i1
  }

  func.func @gt_si16(%a : si16, %b : si16) -> i1 {
    %r = arc.cmpi "gt", %a, %b : si16
    return %r : i1
  }

  func.func @ge_si16(%a : si16, %b : si16) -> i1 {
    %r = arc.cmpi "ge", %a, %b : si16
    return %r : i1
  }

  func.func @eq_si32(%a : si32, %b : si32) -> i1 {
    %r = arc.cmpi "eq", %a, %b : si32
    return %r : i1
  }

  func.func @ne_si32(%a : si32, %b : si32) -> i1 {
    %r = arc.cmpi "ne", %a, %b : si32
    return %r : i1
  }

  func.func @lt_si32(%a : si32, %b : si32) -> i1 {
    %r = arc.cmpi "lt", %a, %b : si32
    return %r : i1
  }

  func.func @le_si32(%a : si32, %b : si32) -> i1 {
    %r = arc.cmpi "le", %a, %b : si32
    return %r : i1
  }

  func.func @gt_si32(%a : si32, %b : si32) -> i1 {
    %r = arc.cmpi "gt", %a, %b : si32
    return %r : i1
  }

  func.func @ge_si32(%a : si32, %b : si32) -> i1 {
    %r = arc.cmpi "ge", %a, %b : si32
    return %r : i1
  }

  func.func @eq_si64(%a : si64, %b : si64) -> i1 {
    %r = arc.cmpi "eq", %a, %b : si64
    return %r : i1
  }

  func.func @ne_si64(%a : si64, %b : si64) -> i1 {
    %r = arc.cmpi "ne", %a, %b : si64
    return %r : i1
  }

  func.func @lt_si64(%a : si64, %b : si64) -> i1 {
    %r = arc.cmpi "lt", %a, %b : si64
    return %r : i1
  }

  func.func @le_si64(%a : si64, %b : si64) -> i1 {
    %r = arc.cmpi "le", %a, %b : si64
    return %r : i1
  }

  func.func @gt_si64(%a : si64, %b : si64) -> i1 {
    %r = arc.cmpi "gt", %a, %b : si64
    return %r : i1
  }

  func.func @ge_si64(%a : si64, %b : si64) -> i1 {
    %r = arc.cmpi "ge", %a, %b : si64
    return %r : i1
  }
}
