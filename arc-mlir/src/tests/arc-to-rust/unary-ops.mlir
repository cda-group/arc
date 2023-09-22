// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorustunaryops {
  func.func @acos_f32(%a : f32) -> f32 {
    %r = arc.acos %a : f32
      return %r : f32
  }

  func.func @asin_f32(%a : f32) -> f32 {
    %r = arc.asin %a : f32
      return %r : f32
  }

  func.func @atan_f32(%a : f32) -> f32 {
    %r = math.atan %a : f32
      return %r : f32
  }

  func.func @cos_f32(%a : f32) -> f32 {
    %r = math.cos %a : f32
      return %r : f32
  }

  func.func @cosh_f32(%a : f32) -> f32 {
    %r = arc.cosh %a : f32
      return %r : f32
  }

  func.func @exp_f32(%a : f32) -> f32 {
    %r = math.exp %a : f32
      return %r : f32
  }

  func.func @log_f32(%a : f32) -> f32 {
    %r = math.log %a : f32
      return %r : f32
  }

  func.func @sin_f32(%a : f32) -> f32 {
    %r = math.sin %a : f32
      return %r : f32
  }

  func.func @sinh_f32(%a : f32) -> f32 {
    %r = arc.sinh %a : f32
      return %r : f32
  }

  func.func @sqrt_f32(%a : f32) -> f32 {
    %r = math.sqrt %a : f32
      return %r : f32
  }

  func.func @tan_f32(%a : f32) -> f32 {
    %r = arc.tan %a : f32
      return %r : f32
  }

  func.func @tanh_f32(%a : f32) -> f32 {
    %r = math.tanh %a : f32
      return %r : f32
  }

  func.func @acos_f64(%a : f64) -> f64 {
    %r = arc.acos %a : f64
      return %r : f64
  }

  func.func @asin_f64(%a : f64) -> f64 {
    %r = arc.asin %a : f64
      return %r : f64
  }

  func.func @atan_f64(%a : f64) -> f64 {
    %r = math.atan %a : f64
      return %r : f64
  }

  func.func @cos_f64(%a : f64) -> f64 {
    %r = math.cos %a : f64
      return %r : f64
  }

  func.func @cosh_f64(%a : f64) -> f64 {
    %r = arc.cosh %a : f64
      return %r : f64
  }

  func.func @exp_f64(%a : f64) -> f64 {
    %r = math.exp %a : f64
      return %r : f64
  }

  func.func @log_f64(%a : f64) -> f64 {
    %r = math.log %a : f64
      return %r : f64
  }

  func.func @sin_f64(%a : f64) -> f64 {
    %r = math.sin %a : f64
      return %r : f64
  }

  func.func @sinh_f64(%a : f64) -> f64 {
    %r = arc.sinh %a : f64
      return %r : f64
  }

  func.func @sqrt_f64(%a : f64) -> f64 {
    %r = math.sqrt %a : f64
      return %r : f64
  }

  func.func @tan_f64(%a : f64) -> f64 {
    %r = arc.tan %a : f64
      return %r : f64
  }

  func.func @tanh_f64(%a : f64) -> f64 {
    %r = math.tanh %a : f64
      return %r : f64
  }
}
