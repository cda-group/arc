// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
module @arctorustunaryops {
func @acos_f32(%a : f32) -> f32 {
  %r = arc.acos %a : f32
  return %r : f32
}

func @asin_f32(%a : f32) -> f32 {
  %r = arc.asin %a : f32
  return %r : f32
}

func @atan_f32(%a : f32) -> f32 {
  %r = math.atan %a : f32
  return %r : f32
}

func @cos_f32(%a : f32) -> f32 {
  %r = math.cos %a : f32
  return %r : f32
}

func @cosh_f32(%a : f32) -> f32 {
  %r = arc.cosh %a : f32
  return %r : f32
}

func @exp_f32(%a : f32) -> f32 {
  %r = math.exp %a : f32
  return %r : f32
}

func @log_f32(%a : f32) -> f32 {
  %r = math.log %a : f32
  return %r : f32
}

func @sin_f32(%a : f32) -> f32 {
  %r = math.sin %a : f32
  return %r : f32
}

func @sinh_f32(%a : f32) -> f32 {
  %r = arc.sinh %a : f32
  return %r : f32
}

func @sqrt_f32(%a : f32) -> f32 {
  %r = math.sqrt %a : f32
  return %r : f32
}

func @tan_f32(%a : f32) -> f32 {
  %r = arc.tan %a : f32
  return %r : f32
}

func @tanh_f32(%a : f32) -> f32 {
  %r = math.tanh %a : f32
  return %r : f32
}

func @acos_f64(%a : f64) -> f64 {
  %r = arc.acos %a : f64
  return %r : f64
}

func @asin_f64(%a : f64) -> f64 {
  %r = arc.asin %a : f64
  return %r : f64
}

func @atan_f64(%a : f64) -> f64 {
  %r = math.atan %a : f64
  return %r : f64
}

func @cos_f64(%a : f64) -> f64 {
  %r = math.cos %a : f64
  return %r : f64
}

func @cosh_f64(%a : f64) -> f64 {
  %r = arc.cosh %a : f64
  return %r : f64
}

func @exp_f64(%a : f64) -> f64 {
  %r = math.exp %a : f64
  return %r : f64
}

func @log_f64(%a : f64) -> f64 {
  %r = math.log %a : f64
  return %r : f64
}

func @sin_f64(%a : f64) -> f64 {
  %r = math.sin %a : f64
  return %r : f64
}

func @sinh_f64(%a : f64) -> f64 {
  %r = arc.sinh %a : f64
  return %r : f64
}

func @sqrt_f64(%a : f64) -> f64 {
  %r = math.sqrt %a : f64
  return %r : f64
}

func @tan_f64(%a : f64) -> f64 {
  %r = arc.tan %a : f64
  return %r : f64
}

func @tanh_f64(%a : f64) -> f64 {
  %r = math.tanh %a : f64
  return %r : f64
}

}
