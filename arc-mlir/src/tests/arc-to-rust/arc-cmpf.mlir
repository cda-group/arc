// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustarcarccmpf {
func @oeq_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "oeq", %a, %b : f32
  return %r : i1
}

func @one_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "one", %a, %b : f32
  return %r : i1
}

func @olt_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "olt", %a, %b : f32
  return %r : i1
}

func @ole_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "ole", %a, %b : f32
  return %r : i1
}

func @ogt_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "ogt", %a, %b : f32
  return %r : i1
}

func @oge_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "oge", %a, %b : f32
  return %r : i1
}

func @oeq_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "oeq", %a, %b : f64
  return %r : i1
}

func @one_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "one", %a, %b : f64
  return %r : i1
}

func @olt_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "olt", %a, %b : f64
  return %r : i1
}

func @ole_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "ole", %a, %b : f64
  return %r : i1
}

func @ogt_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "ogt", %a, %b : f64
  return %r : i1
}

func @oge_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "oge", %a, %b : f64
  return %r : i1
}

}
