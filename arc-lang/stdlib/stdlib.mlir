// Integer arithmetics

func @add_u8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.addi %a, %b : ui8
  return %c : ui8
}

func @sub_u8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.subi %a, %b : ui8
  return %c : ui8
}

func @mul_u8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.muli %a, %b : ui8
  return %c : ui8
}

func @div_u8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.divi %a, %b : ui8
  return %c : ui8
}

func @rem_u8(%a : ui8, %b : ui8) -> ui8 {
  %c = arc.remi %a, %b : ui8
  return %c : ui8
}

func @add_u16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.addi %a, %b : ui16
  return %c : ui16
}

func @sub_u16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.subi %a, %b : ui16
  return %c : ui16
}

func @mul_u16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.muli %a, %b : ui16
  return %c : ui16
}

func @div_u16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.divi %a, %b : ui16
  return %c : ui16
}

func @rem_u16(%a : ui16, %b : ui16) -> ui16 {
  %c = arc.remi %a, %b : ui16
  return %c : ui16
}

func @add_u32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.addi %a, %b : ui32
  return %c : ui32
}

func @sub_u32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.subi %a, %b : ui32
  return %c : ui32
}

func @mul_u32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.muli %a, %b : ui32
  return %c : ui32
}

func @div_u32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.divi %a, %b : ui32
  return %c : ui32
}

func @rem_u32(%a : ui32, %b : ui32) -> ui32 {
  %c = arc.remi %a, %b : ui32
  return %c : ui32
}

func @add_u64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.addi %a, %b : ui64
  return %c : ui64
}

func @sub_u64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.subi %a, %b : ui64
  return %c : ui64
}

func @mul_u64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.muli %a, %b : ui64
  return %c : ui64
}

func @div_u64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.divi %a, %b : ui64
  return %c : ui64
}

func @rem_u64(%a : ui64, %b : ui64) -> ui64 {
  %c = arc.remi %a, %b : ui64
  return %c : ui64
}

func @add_i8(%a : si8, %b : si8) -> si8 {
  %c = arc.addi %a, %b : si8
  return %c : si8
}

func @sub_i8(%a : si8, %b : si8) -> si8 {
  %c = arc.subi %a, %b : si8
  return %c : si8
}

func @mul_i8(%a : si8, %b : si8) -> si8 {
  %c = arc.muli %a, %b : si8
  return %c : si8
}

func @div_i8(%a : si8, %b : si8) -> si8 {
  %c = arc.divi %a, %b : si8
  return %c : si8
}

func @rem_i8(%a : si8, %b : si8) -> si8 {
  %c = arc.remi %a, %b : si8
  return %c : si8
}

func @add_i16(%a : si16, %b : si16) -> si16 {
  %c = arc.addi %a, %b : si16
  return %c : si16
}

func @sub_i16(%a : si16, %b : si16) -> si16 {
  %c = arc.subi %a, %b : si16
  return %c : si16
}

func @mul_i16(%a : si16, %b : si16) -> si16 {
  %c = arc.muli %a, %b : si16
  return %c : si16
}

func @div_i16(%a : si16, %b : si16) -> si16 {
  %c = arc.divi %a, %b : si16
  return %c : si16
}

func @rem_i16(%a : si16, %b : si16) -> si16 {
  %c = arc.remi %a, %b : si16
  return %c : si16
}

func @add_i32(%a : si32, %b : si32) -> si32 {
  %c = arc.addi %a, %b : si32
  return %c : si32
}

func @sub_i32(%a : si32, %b : si32) -> si32 {
  %c = arc.subi %a, %b : si32
  return %c : si32
}

func @mul_i32(%a : si32, %b : si32) -> si32 {
  %c = arc.muli %a, %b : si32
  return %c : si32
}

func @div_i32(%a : si32, %b : si32) -> si32 {
  %c = arc.divi %a, %b : si32
  return %c : si32
}

func @rem_i32(%a : si32, %b : si32) -> si32 {
  %c = arc.remi %a, %b : si32
  return %c : si32
}

func @add_i64(%a : si64, %b : si64) -> si64 {
  %c = arc.addi %a, %b : si64
  return %c : si64
}

func @sub_i64(%a : si64, %b : si64) -> si64 {
  %c = arc.subi %a, %b : si64
  return %c : si64
}

func @mul_i64(%a : si64, %b : si64) -> si64 {
  %c = arc.muli %a, %b : si64
  return %c : si64
}

func @div_i64(%a : si64, %b : si64) -> si64 {
  %c = arc.divi %a, %b : si64
  return %c : si64
}

func @rem_i64(%a : si64, %b : si64) -> si64 {
  %c = arc.remi %a, %b : si64
  return %c : si64
}

// Float arithmetics

func @rem_f32(%a : f32, %b : f32) -> f32 {
  %c = arith.remf %a, %b : f32
  return %c : f32
}

func @add_f32(%a: f32, %b: f32) -> f32 {
  %c = arith.addf %a, %b : f32
  return %c : f32
}

func @sub_f32(%a: f32, %b: f32) -> f32 {
  %c = arith.subf %a, %b : f32
  return %c : f32
}

func @mul_f32(%a: f32, %b: f32) -> f32 {
  %c = arith.mulf %a, %b : f32
  return %c : f32
}

func @div_f32(%a: f32, %b: f32) -> f32 {
  %c = arith.divf %a, %b : f32
  return %c : f32
}

func @pow_f32(%a: f32, %b: f32) -> f32 {
  %c = math.powf %a, %b : f32
  return %c : f32
}

func @rem_f64(%a : f64, %b : f64) -> f64 {
  %c = arith.remf %a, %b : f64
  return %c : f64
}

func @add_f64(%a: f64, %b: f64) -> f64 {
  %c = arith.addf %a, %b : f64
  return %c : f64
}

func @sub_f64(%a: f64, %b: f64) -> f64 {
  %c = arith.subf %a, %b : f64
  return %c : f64
}

func @mul_f64(%a: f64, %b: f64) -> f64 {
  %c = arith.mulf %a, %b : f64
  return %c : f64
}

func @div_f64(%a: f64, %b: f64) -> f64 {
  %c = arith.divf %a, %b : f64
  return %c : f64
}

func @pow_f64(%a : f64, %b : f64) -> f64 {
  %c = math.powf %a, %b : f64
  return %c : f64
}

// Logical operations

func @and_i1(%arg0: i1, %arg1: i1) -> i1 {
  %0 = arith.andi %arg0, %arg1 : i1
  return %0 : i1
}

func @or_i1(%arg0: i1, %arg1: i1) -> i1 {
  %0 = arith.ori %arg0, %arg1 : i1
  return %0 : i1
}

func @xor_i1(%arg0: i1, %arg1: i1) -> i1 {
  %0 = arith.xori %arg0, %arg1 : i1
  return %0 : i1
}

func @and_i8(%arg0: i8, %arg1: i8) -> i8 {
  %0 = arith.andi %arg0, %arg1 : i8
  return %0 : i8
}

func @or_i8(%arg0: i8, %arg1: i8) -> i8 {
  %0 = arith.ori %arg0, %arg1 : i8
  return %0 : i8
}

func @xor_i8(%arg0: i8, %arg1: i8) -> i8 {
  %0 = arith.xori %arg0, %arg1 : i8
  return %0 : i8
}

func @and_i16(%arg0: i16, %arg1: i16) -> i16 {
  %0 = arith.andi %arg0, %arg1 : i16
  return %0 : i16
}

func @or_i16(%arg0: i16, %arg1: i16) -> i16 {
  %0 = arith.ori %arg0, %arg1 : i16
  return %0 : i16
}

func @xor_i16(%arg0: i16, %arg1: i16) -> i16 {
  %0 = arith.xori %arg0, %arg1 : i16
  return %0 : i16
}

func @and_i32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.andi %arg0, %arg1 : i32
  return %0 : i32
}

func @or_i32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.ori %arg0, %arg1 : i32
  return %0 : i32
}

func @xor_i32(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.xori %arg0, %arg1 : i32
  return %0 : i32
}

func @and_i64(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.andi %arg0, %arg1 : i64
  return %0 : i64
}

func @or_i64(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.ori %arg0, %arg1 : i64
  return %0 : i64
}

func @xor_i64(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.xori %arg0, %arg1 : i64
  return %0 : i64
}

// Comparison

func @eq_u8(%a : ui8, %b : ui8) -> i1 {
  %r = arc.cmpi "eq", %a, %b : ui8
  return %r : i1
}

func @ne_u8(%a : ui8, %b : ui8) -> i1 {
  %r = arc.cmpi "ne", %a, %b : ui8
  return %r : i1
}

func @lt_u8(%a : ui8, %b : ui8) -> i1 {
  %r = arc.cmpi "lt", %a, %b : ui8
  return %r : i1
}

func @le_u8(%a : ui8, %b : ui8) -> i1 {
  %r = arc.cmpi "le", %a, %b : ui8
  return %r : i1
}

func @gt_u8(%a : ui8, %b : ui8) -> i1 {
  %r = arc.cmpi "gt", %a, %b : ui8
  return %r : i1
}

func @ge_u8(%a : ui8, %b : ui8) -> i1 {
  %r = arc.cmpi "ge", %a, %b : ui8
  return %r : i1
}

func @eq_u16(%a : ui16, %b : ui16) -> i1 {
  %r = arc.cmpi "eq", %a, %b : ui16
  return %r : i1
}

func @ne_u16(%a : ui16, %b : ui16) -> i1 {
  %r = arc.cmpi "ne", %a, %b : ui16
  return %r : i1
}

func @lt_u16(%a : ui16, %b : ui16) -> i1 {
  %r = arc.cmpi "lt", %a, %b : ui16
  return %r : i1
}

func @le_u16(%a : ui16, %b : ui16) -> i1 {
  %r = arc.cmpi "le", %a, %b : ui16
  return %r : i1
}

func @gt_u16(%a : ui16, %b : ui16) -> i1 {
  %r = arc.cmpi "gt", %a, %b : ui16
  return %r : i1
}

func @ge_u16(%a : ui16, %b : ui16) -> i1 {
  %r = arc.cmpi "ge", %a, %b : ui16
  return %r : i1
}

func @eq_u32(%a : ui32, %b : ui32) -> i1 {
  %r = arc.cmpi "eq", %a, %b : ui32
  return %r : i1
}

func @ne_u32(%a : ui32, %b : ui32) -> i1 {
  %r = arc.cmpi "ne", %a, %b : ui32
  return %r : i1
}

func @lt_u32(%a : ui32, %b : ui32) -> i1 {
  %r = arc.cmpi "lt", %a, %b : ui32
  return %r : i1
}

func @le_u32(%a : ui32, %b : ui32) -> i1 {
  %r = arc.cmpi "le", %a, %b : ui32
  return %r : i1
}

func @gt_u32(%a : ui32, %b : ui32) -> i1 {
  %r = arc.cmpi "gt", %a, %b : ui32
  return %r : i1
}

func @ge_u32(%a : ui32, %b : ui32) -> i1 {
  %r = arc.cmpi "ge", %a, %b : ui32
  return %r : i1
}

func @eq_u64(%a : ui64, %b : ui64) -> i1 {
  %r = arc.cmpi "eq", %a, %b : ui64
  return %r : i1
}

func @ne_u64(%a : ui64, %b : ui64) -> i1 {
  %r = arc.cmpi "ne", %a, %b : ui64
  return %r : i1
}

func @lt_u64(%a : ui64, %b : ui64) -> i1 {
  %r = arc.cmpi "lt", %a, %b : ui64
  return %r : i1
}

func @le_u64(%a : ui64, %b : ui64) -> i1 {
  %r = arc.cmpi "le", %a, %b : ui64
  return %r : i1
}

func @gt_u64(%a : ui64, %b : ui64) -> i1 {
  %r = arc.cmpi "gt", %a, %b : ui64
  return %r : i1
}

func @ge_u64(%a : ui64, %b : ui64) -> i1 {
  %r = arc.cmpi "ge", %a, %b : ui64
  return %r : i1
}

func @eq_i8(%a : si8, %b : si8) -> i1 {
  %r = arc.cmpi "eq", %a, %b : si8
  return %r : i1
}

func @ne_i8(%a : si8, %b : si8) -> i1 {
  %r = arc.cmpi "ne", %a, %b : si8
  return %r : i1
}

func @lt_i8(%a : si8, %b : si8) -> i1 {
  %r = arc.cmpi "lt", %a, %b : si8
  return %r : i1
}

func @le_i8(%a : si8, %b : si8) -> i1 {
  %r = arc.cmpi "le", %a, %b : si8
  return %r : i1
}

func @gt_i8(%a : si8, %b : si8) -> i1 {
  %r = arc.cmpi "gt", %a, %b : si8
  return %r : i1
}

func @ge_i8(%a : si8, %b : si8) -> i1 {
  %r = arc.cmpi "ge", %a, %b : si8
  return %r : i1
}

func @eq_i16(%a : si16, %b : si16) -> i1 {
  %r = arc.cmpi "eq", %a, %b : si16
  return %r : i1
}

func @ne_i16(%a : si16, %b : si16) -> i1 {
  %r = arc.cmpi "ne", %a, %b : si16
  return %r : i1
}

func @lt_i16(%a : si16, %b : si16) -> i1 {
  %r = arc.cmpi "lt", %a, %b : si16
  return %r : i1
}

func @le_i16(%a : si16, %b : si16) -> i1 {
  %r = arc.cmpi "le", %a, %b : si16
  return %r : i1
}

func @gt_i16(%a : si16, %b : si16) -> i1 {
  %r = arc.cmpi "gt", %a, %b : si16
  return %r : i1
}

func @ge_i16(%a : si16, %b : si16) -> i1 {
  %r = arc.cmpi "ge", %a, %b : si16
  return %r : i1
}

func @eq_i32(%a : si32, %b : si32) -> i1 {
  %r = arc.cmpi "eq", %a, %b : si32
  return %r : i1
}

func @ne_i32(%a : si32, %b : si32) -> i1 {
  %r = arc.cmpi "ne", %a, %b : si32
  return %r : i1
}

func @lt_i32(%a : si32, %b : si32) -> i1 {
  %r = arc.cmpi "lt", %a, %b : si32
  return %r : i1
}

func @le_i32(%a : si32, %b : si32) -> i1 {
  %r = arc.cmpi "le", %a, %b : si32
  return %r : i1
}

func @gt_i32(%a : si32, %b : si32) -> i1 {
  %r = arc.cmpi "gt", %a, %b : si32
  return %r : i1
}

func @ge_i32(%a : si32, %b : si32) -> i1 {
  %r = arc.cmpi "ge", %a, %b : si32
  return %r : i1
}

func @eq_i64(%a : si64, %b : si64) -> i1 {
  %r = arc.cmpi "eq", %a, %b : si64
  return %r : i1
}

func @ne_i64(%a : si64, %b : si64) -> i1 {
  %r = arc.cmpi "ne", %a, %b : si64
  return %r : i1
}

func @lt_i64(%a : si64, %b : si64) -> i1 {
  %r = arc.cmpi "lt", %a, %b : si64
  return %r : i1
}

func @le_i64(%a : si64, %b : si64) -> i1 {
  %r = arc.cmpi "le", %a, %b : si64
  return %r : i1
}

func @gt_i64(%a : si64, %b : si64) -> i1 {
  %r = arc.cmpi "gt", %a, %b : si64
  return %r : i1
}

func @ge_i64(%a : si64, %b : si64) -> i1 {
  %r = arc.cmpi "ge", %a, %b : si64
  return %r : i1
}

// ---------------------------------------

func @eq_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "oeq", %a, %b : f32
  return %r : i1
}

func @ne_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "one", %a, %b : f32
  return %r : i1
}

func @lt_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "olt", %a, %b : f32
  return %r : i1
}

func @le_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "ole", %a, %b : f32
  return %r : i1
}

func @gt_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "ogt", %a, %b : f32
  return %r : i1
}

func @ge_f32(%a : f32, %b : f32) -> i1 {
  %r = arith.cmpf "oge", %a, %b : f32
  return %r : i1
}

func @eq_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "oeq", %a, %b : f64
  return %r : i1
}

func @ne_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "one", %a, %b : f64
  return %r : i1
}

func @lt_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "olt", %a, %b : f64
  return %r : i1
}

func @le_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "ole", %a, %b : f64
  return %r : i1
}

func @gt_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "ogt", %a, %b : f64
  return %r : i1
}

func @ge_f64(%a : f64, %b : f64) -> i1 {
  %r = arith.cmpf "oge", %a, %b : f64
  return %r : i1
}

// Unary ops

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
