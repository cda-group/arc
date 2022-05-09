// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustintarith {
func.func @addi_ui8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.addi %a, %b : ui8
  return %c : ui8
}

func.func @subi_ui8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.subi %a, %b : ui8
  return %c : ui8
}

func.func @muli_ui8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.muli %a, %b : ui8
  return %c : ui8
}

func.func @divi_ui8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.divi %a, %b : ui8
  return %c : ui8
}

func.func @remi_ui8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.remi %a, %b : ui8
  return %c : ui8
}

func.func @addi_ui16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.addi %a, %b : ui16
  return %c : ui16
}

func.func @subi_ui16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.subi %a, %b : ui16
  return %c : ui16
}

func.func @muli_ui16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.muli %a, %b : ui16
  return %c : ui16
}

func.func @divi_ui16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.divi %a, %b : ui16
  return %c : ui16
}

func.func @remi_ui16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.remi %a, %b : ui16
  return %c : ui16
}

func.func @addi_ui32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.addi %a, %b : ui32
  return %c : ui32
}

func.func @subi_ui32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.subi %a, %b : ui32
  return %c : ui32
}

func.func @muli_ui32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.muli %a, %b : ui32
  return %c : ui32
}

func.func @divi_ui32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.divi %a, %b : ui32
  return %c : ui32
}

func.func @remi_ui32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.remi %a, %b : ui32
  return %c : ui32
}

func.func @addi_ui64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.addi %a, %b : ui64
  return %c : ui64
}

func.func @subi_ui64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.subi %a, %b : ui64
  return %c : ui64
}

func.func @muli_ui64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.muli %a, %b : ui64
  return %c : ui64
}

func.func @divi_ui64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.divi %a, %b : ui64
  return %c : ui64
}

func.func @remi_ui64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.remi %a, %b : ui64
  return %c : ui64
}

func.func @addi_si8(%a : si8, %b : si8) -> si8 {
  %c = arc.addi %a, %b : si8
  return %c : si8
}

func.func @subi_si8(%a : si8, %b : si8) -> si8 {
  %c = arc.subi %a, %b : si8
  return %c : si8
}

func.func @muli_si8(%a : si8, %b : si8) -> si8 {
  %c = arc.muli %a, %b : si8
  return %c : si8
}

func.func @divi_si8(%a : si8, %b : si8) -> si8 {
  %c = arc.divi %a, %b : si8
  return %c : si8
}

func.func @remi_si8(%a : si8, %b : si8) -> si8 {
  %c = arc.remi %a, %b : si8
  return %c : si8
}

func.func @addi_si16(%a : si16, %b : si16) -> si16 {
  %c = arc.addi %a, %b : si16
  return %c : si16
}

func.func @subi_si16(%a : si16, %b : si16) -> si16 {
  %c = arc.subi %a, %b : si16
  return %c : si16
}

func.func @muli_si16(%a : si16, %b : si16) -> si16 {
  %c = arc.muli %a, %b : si16
  return %c : si16
}

func.func @divi_si16(%a : si16, %b : si16) -> si16 {
  %c = arc.divi %a, %b : si16
  return %c : si16
}

func.func @remi_si16(%a : si16, %b : si16) -> si16 {
  %c = arc.remi %a, %b : si16
  return %c : si16
}

func.func @addi_si32(%a : si32, %b : si32) -> si32 {
  %c = arc.addi %a, %b : si32
  return %c : si32
}

func.func @subi_si32(%a : si32, %b : si32) -> si32 {
  %c = arc.subi %a, %b : si32
  return %c : si32
}

func.func @muli_si32(%a : si32, %b : si32) -> si32 {
  %c = arc.muli %a, %b : si32
  return %c : si32
}

func.func @divi_si32(%a : si32, %b : si32) -> si32 {
  %c = arc.divi %a, %b : si32
  return %c : si32
}

func.func @remi_si32(%a : si32, %b : si32) -> si32 {
  %c = arc.remi %a, %b : si32
  return %c : si32
}

func.func @addi_si64(%a : si64, %b : si64) -> si64 {
  %c = arc.addi %a, %b : si64
  return %c : si64
}

func.func @subi_si64(%a : si64, %b : si64) -> si64 {
  %c = arc.subi %a, %b : si64
  return %c : si64
}

func.func @muli_si64(%a : si64, %b : si64) -> si64 {
  %c = arc.muli %a, %b : si64
  return %c : si64
}

func.func @divi_si64(%a : si64, %b : si64) -> si64 {
  %c = arc.divi %a, %b : si64
  return %c : si64
}

func.func @remi_si64(%a : si64, %b : si64) -> si64 {
  %c = arc.remi %a, %b : si64
  return %c : si64
}

}
