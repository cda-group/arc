// Integer arithmetics

func.func @add_u8(%a : ui8, %b : ui8) -> ui8 attributes {rust.declare} {
  %c = arc.addi %a, %b : ui8
  return %c : ui8
}

func.func @sub_u8(%a : ui8, %b : ui8) -> ui8 attributes {rust.declare} {
  %c = arc.subi %a, %b : ui8
  return %c : ui8
}

func.func @mul_u8(%a : ui8, %b : ui8) -> ui8 attributes {rust.declare} {
  %c = arc.muli %a, %b : ui8
  return %c : ui8
}

func.func @div_u8(%a : ui8, %b : ui8) -> ui8 attributes {rust.declare} {
  %c = arc.divi %a, %b : ui8
  return %c : ui8
}

func.func @rem_u8(%a : ui8, %b : ui8) -> ui8 attributes {rust.declare} {
  %c = arc.remi %a, %b : ui8
  return %c : ui8
}

func.func @add_u16(%a : ui16, %b : ui16) -> ui16 attributes {rust.declare} {
  %c = arc.addi %a, %b : ui16
  return %c : ui16
}

func.func @sub_u16(%a : ui16, %b : ui16) -> ui16 attributes {rust.declare} {
  %c = arc.subi %a, %b : ui16
  return %c : ui16
}

func.func @mul_u16(%a : ui16, %b : ui16) -> ui16 attributes {rust.declare} {
  %c = arc.muli %a, %b : ui16
  return %c : ui16
}

func.func @div_u16(%a : ui16, %b : ui16) -> ui16 attributes {rust.declare} {
  %c = arc.divi %a, %b : ui16
  return %c : ui16
}

func.func @rem_u16(%a : ui16, %b : ui16) -> ui16 attributes {rust.declare} {
  %c = arc.remi %a, %b : ui16
  return %c : ui16
}

func.func @add_u32(%a : ui32, %b : ui32) -> ui32 attributes {rust.declare} {
  %c = arc.addi %a, %b : ui32
  return %c : ui32
}

func.func @sub_u32(%a : ui32, %b : ui32) -> ui32 attributes {rust.declare} {
  %c = arc.subi %a, %b : ui32
  return %c : ui32
}

func.func @mul_u32(%a : ui32, %b : ui32) -> ui32 attributes {rust.declare} {
  %c = arc.muli %a, %b : ui32
  return %c : ui32
}

func.func @div_u32(%a : ui32, %b : ui32) -> ui32 attributes {rust.declare} {
  %c = arc.divi %a, %b : ui32
  return %c : ui32
}

func.func @rem_u32(%a : ui32, %b : ui32) -> ui32 attributes {rust.declare} {
  %c = arc.remi %a, %b : ui32
  return %c : ui32
}

func.func @add_u64(%a : ui64, %b : ui64) -> ui64 attributes {rust.declare} {
  %c = arc.addi %a, %b : ui64
  return %c : ui64
}

func.func @sub_u64(%a : ui64, %b : ui64) -> ui64 attributes {rust.declare} {
  %c = arc.subi %a, %b : ui64
  return %c : ui64
}

func.func @mul_u64(%a : ui64, %b : ui64) -> ui64 attributes {rust.declare} {
  %c = arc.muli %a, %b : ui64
  return %c : ui64
}

func.func @div_u64(%a : ui64, %b : ui64) -> ui64 attributes {rust.declare} {
  %c = arc.divi %a, %b : ui64
  return %c : ui64
}

func.func @rem_u64(%a : ui64, %b : ui64) -> ui64 attributes {rust.declare} {
  %c = arc.remi %a, %b : ui64
  return %c : ui64
}

func.func @add_i8(%a : si8, %b : si8) -> si8 attributes {rust.declare} {
  %c = arc.addi %a, %b : si8
  return %c : si8
}

func.func @sub_i8(%a : si8, %b : si8) -> si8 attributes {rust.declare} {
  %c = arc.subi %a, %b : si8
  return %c : si8
}

func.func @mul_i8(%a : si8, %b : si8) -> si8 attributes {rust.declare} {
  %c = arc.muli %a, %b : si8
  return %c : si8
}

func.func @div_i8(%a : si8, %b : si8) -> si8 attributes {rust.declare} {
  %c = arc.divi %a, %b : si8
  return %c : si8
}

func.func @rem_i8(%a : si8, %b : si8) -> si8 attributes {rust.declare} {
  %c = arc.remi %a, %b : si8
  return %c : si8
}

func.func @add_i16(%a : si16, %b : si16) -> si16 attributes {rust.declare} {
  %c = arc.addi %a, %b : si16
  return %c : si16
}

func.func @sub_i16(%a : si16, %b : si16) -> si16 attributes {rust.declare} {
  %c = arc.subi %a, %b : si16
  return %c : si16
}

func.func @mul_i16(%a : si16, %b : si16) -> si16 attributes {rust.declare} {
  %c = arc.muli %a, %b : si16
  return %c : si16
}

func.func @div_i16(%a : si16, %b : si16) -> si16 attributes {rust.declare} {
  %c = arc.divi %a, %b : si16
  return %c : si16
}

func.func @rem_i16(%a : si16, %b : si16) -> si16 attributes {rust.declare} {
  %c = arc.remi %a, %b : si16
  return %c : si16
}

func.func @add_i32(%a : si32, %b : si32) -> si32 attributes {rust.declare} {
  %c = arc.addi %a, %b : si32
  return %c : si32
}

func.func @sub_i32(%a : si32, %b : si32) -> si32 attributes {rust.declare} {
  %c = arc.subi %a, %b : si32
  return %c : si32
}

func.func @mul_i32(%a : si32, %b : si32) -> si32 attributes {rust.declare} {
  %c = arc.muli %a, %b : si32
  return %c : si32
}

func.func @div_i32(%a : si32, %b : si32) -> si32 attributes {rust.declare} {
  %c = arc.divi %a, %b : si32
  return %c : si32
}

func.func @rem_i32(%a : si32, %b : si32) -> si32 attributes {rust.declare} {
  %c = arc.remi %a, %b : si32
  return %c : si32
}

func.func @add_i64(%a : si64, %b : si64) -> si64 attributes {rust.declare} {
  %c = arc.addi %a, %b : si64
  return %c : si64
}

func.func @sub_i64(%a : si64, %b : si64) -> si64 attributes {rust.declare} {
  %c = arc.subi %a, %b : si64
  return %c : si64
}

func.func @mul_i64(%a : si64, %b : si64) -> si64 attributes {rust.declare} {
  %c = arc.muli %a, %b : si64
  return %c : si64
}

func.func @div_i64(%a : si64, %b : si64) -> si64 attributes {rust.declare} {
  %c = arc.divi %a, %b : si64
  return %c : si64
}

func.func @rem_i64(%a : si64, %b : si64) -> si64 attributes {rust.declare} {
  %c = arc.remi %a, %b : si64
  return %c : si64
}

// Float arithmetics

func.func @rem_f32(%a : f32, %b : f32) -> f32 attributes {rust.declare} {
  %c = arith.remf %a, %b : f32
  return %c : f32
}

func.func @add_f32(%a: f32, %b: f32) -> f32 attributes {rust.declare} {
  %c = arith.addf %a, %b : f32
  return %c : f32
}

func.func @sub_f32(%a: f32, %b: f32) -> f32 attributes {rust.declare} {
  %c = arith.subf %a, %b : f32
  return %c : f32
}

func.func @mul_f32(%a: f32, %b: f32) -> f32 attributes {rust.declare} {
  %c = arith.mulf %a, %b : f32
  return %c : f32
}

func.func @div_f32(%a: f32, %b: f32) -> f32 attributes {rust.declare} {
  %c = arith.divf %a, %b : f32
  return %c : f32
}

func.func @pow_f32(%a: f32, %b: f32) -> f32 attributes {rust.declare} {
  %c = math.powf %a, %b : f32
  return %c : f32
}

func.func @rem_f64(%a : f64, %b : f64) -> f64 attributes {rust.declare} {
  %c = arith.remf %a, %b : f64
  return %c : f64
}

func.func @add_f64(%a: f64, %b: f64) -> f64 attributes {rust.declare} {
  %c = arith.addf %a, %b : f64
  return %c : f64
}

func.func @sub_f64(%a: f64, %b: f64) -> f64 attributes {rust.declare} {
  %c = arith.subf %a, %b : f64
  return %c : f64
}

func.func @mul_f64(%a: f64, %b: f64) -> f64 attributes {rust.declare} {
  %c = arith.mulf %a, %b : f64
  return %c : f64
}

func.func @div_f64(%a: f64, %b: f64) -> f64 attributes {rust.declare} {
  %c = arith.divf %a, %b : f64
  return %c : f64
}

func.func @pow_f64(%a : f64, %b : f64) -> f64 attributes {rust.declare} {
  %c = math.powf %a, %b : f64
  return %c : f64
}

// Logical operations

func.func @eq_i1(%a : i1, %b : i1) -> i1 attributes {rust.declare} {
  %r = arith.cmpi "eq", %a, %b : i1
  return %r : i1
}

func.func @and_i1(%arg0: i1, %arg1: i1) -> i1 attributes {rust.declare} {
  %0 = arith.andi %arg0, %arg1 : i1
  return %0 : i1
}

func.func @or_i1(%arg0: i1, %arg1: i1) -> i1 attributes {rust.declare} {
  %0 = arith.ori %arg0, %arg1 : i1
  return %0 : i1
}

func.func @xor_i1(%arg0: i1, %arg1: i1) -> i1 attributes {rust.declare} {
  %0 = arith.xori %arg0, %arg1 : i1
  return %0 : i1
}

func.func @and_i8(%arg0: i8, %arg1: i8) -> i8 attributes {rust.declare} {
  %0 = arith.andi %arg0, %arg1 : i8
  return %0 : i8
}

func.func @or_i8(%arg0: i8, %arg1: i8) -> i8 attributes {rust.declare} {
  %0 = arith.ori %arg0, %arg1 : i8
  return %0 : i8
}

func.func @xor_i8(%arg0: i8, %arg1: i8) -> i8 attributes {rust.declare} {
  %0 = arith.xori %arg0, %arg1 : i8
  return %0 : i8
}

func.func @and_i16(%arg0: i16, %arg1: i16) -> i16 attributes {rust.declare} {
  %0 = arith.andi %arg0, %arg1 : i16
  return %0 : i16
}

func.func @or_i16(%arg0: i16, %arg1: i16) -> i16 attributes {rust.declare} {
  %0 = arith.ori %arg0, %arg1 : i16
  return %0 : i16
}

func.func @xor_i16(%arg0: i16, %arg1: i16) -> i16 attributes {rust.declare} {
  %0 = arith.xori %arg0, %arg1 : i16
  return %0 : i16
}

func.func @and_i32(%arg0: i32, %arg1: i32) -> i32 attributes {rust.declare} {
  %0 = arith.andi %arg0, %arg1 : i32
  return %0 : i32
}

func.func @or_i32(%arg0: i32, %arg1: i32) -> i32 attributes {rust.declare} {
  %0 = arith.ori %arg0, %arg1 : i32
  return %0 : i32
}

func.func @xor_i32(%arg0: i32, %arg1: i32) -> i32 attributes {rust.declare} {
  %0 = arith.xori %arg0, %arg1 : i32
  return %0 : i32
}

func.func @and_i64(%arg0: i64, %arg1: i64) -> i64 attributes {rust.declare} {
  %0 = arith.andi %arg0, %arg1 : i64
  return %0 : i64
}

func.func @or_i64(%arg0: i64, %arg1: i64) -> i64 attributes {rust.declare} {
  %0 = arith.ori %arg0, %arg1 : i64
  return %0 : i64
}

func.func @xor_i64(%arg0: i64, %arg1: i64) -> i64 attributes {rust.declare} {
  %0 = arith.xori %arg0, %arg1 : i64
  return %0 : i64
}

// Comparison

func.func @eq_u8(%a : ui8, %b : ui8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "eq", %a, %b : ui8
  return %r : i1
}

func.func @ne_u8(%a : ui8, %b : ui8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ne", %a, %b : ui8
  return %r : i1
}

func.func @lt_u8(%a : ui8, %b : ui8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "lt", %a, %b : ui8
  return %r : i1
}

func.func @le_u8(%a : ui8, %b : ui8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "le", %a, %b : ui8
  return %r : i1
}

func.func @gt_u8(%a : ui8, %b : ui8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "gt", %a, %b : ui8
  return %r : i1
}

func.func @ge_u8(%a : ui8, %b : ui8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ge", %a, %b : ui8
  return %r : i1
}

func.func @eq_u16(%a : ui16, %b : ui16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "eq", %a, %b : ui16
  return %r : i1
}

func.func @ne_u16(%a : ui16, %b : ui16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ne", %a, %b : ui16
  return %r : i1
}

func.func @lt_u16(%a : ui16, %b : ui16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "lt", %a, %b : ui16
  return %r : i1
}

func.func @le_u16(%a : ui16, %b : ui16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "le", %a, %b : ui16
  return %r : i1
}

func.func @gt_u16(%a : ui16, %b : ui16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "gt", %a, %b : ui16
  return %r : i1
}

func.func @ge_u16(%a : ui16, %b : ui16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ge", %a, %b : ui16
  return %r : i1
}

func.func @eq_u32(%a : ui32, %b : ui32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "eq", %a, %b : ui32
  return %r : i1
}

func.func @ne_u32(%a : ui32, %b : ui32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ne", %a, %b : ui32
  return %r : i1
}

func.func @lt_u32(%a : ui32, %b : ui32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "lt", %a, %b : ui32
  return %r : i1
}

func.func @le_u32(%a : ui32, %b : ui32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "le", %a, %b : ui32
  return %r : i1
}

func.func @gt_u32(%a : ui32, %b : ui32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "gt", %a, %b : ui32
  return %r : i1
}

func.func @ge_u32(%a : ui32, %b : ui32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ge", %a, %b : ui32
  return %r : i1
}

func.func @eq_u64(%a : ui64, %b : ui64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "eq", %a, %b : ui64
  return %r : i1
}

func.func @ne_u64(%a : ui64, %b : ui64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ne", %a, %b : ui64
  return %r : i1
}

func.func @lt_u64(%a : ui64, %b : ui64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "lt", %a, %b : ui64
  return %r : i1
}

func.func @le_u64(%a : ui64, %b : ui64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "le", %a, %b : ui64
  return %r : i1
}

func.func @gt_u64(%a : ui64, %b : ui64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "gt", %a, %b : ui64
  return %r : i1
}

func.func @ge_u64(%a : ui64, %b : ui64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ge", %a, %b : ui64
  return %r : i1
}

func.func @eq_i8(%a : si8, %b : si8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "eq", %a, %b : si8
  return %r : i1
}

func.func @ne_i8(%a : si8, %b : si8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ne", %a, %b : si8
  return %r : i1
}

func.func @lt_i8(%a : si8, %b : si8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "lt", %a, %b : si8
  return %r : i1
}

func.func @le_i8(%a : si8, %b : si8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "le", %a, %b : si8
  return %r : i1
}

func.func @gt_i8(%a : si8, %b : si8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "gt", %a, %b : si8
  return %r : i1
}

func.func @ge_i8(%a : si8, %b : si8) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ge", %a, %b : si8
  return %r : i1
}

func.func @eq_i16(%a : si16, %b : si16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "eq", %a, %b : si16
  return %r : i1
}

func.func @ne_i16(%a : si16, %b : si16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ne", %a, %b : si16
  return %r : i1
}

func.func @lt_i16(%a : si16, %b : si16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "lt", %a, %b : si16
  return %r : i1
}

func.func @le_i16(%a : si16, %b : si16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "le", %a, %b : si16
  return %r : i1
}

func.func @gt_i16(%a : si16, %b : si16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "gt", %a, %b : si16
  return %r : i1
}

func.func @ge_i16(%a : si16, %b : si16) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ge", %a, %b : si16
  return %r : i1
}

func.func @eq_i32(%a : si32, %b : si32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "eq", %a, %b : si32
  return %r : i1
}

func.func @ne_i32(%a : si32, %b : si32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ne", %a, %b : si32
  return %r : i1
}

func.func @lt_i32(%a : si32, %b : si32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "lt", %a, %b : si32
  return %r : i1
}

func.func @le_i32(%a : si32, %b : si32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "le", %a, %b : si32
  return %r : i1
}

func.func @gt_i32(%a : si32, %b : si32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "gt", %a, %b : si32
  return %r : i1
}

func.func @ge_i32(%a : si32, %b : si32) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ge", %a, %b : si32
  return %r : i1
}

func.func @eq_i64(%a : si64, %b : si64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "eq", %a, %b : si64
  return %r : i1
}

func.func @ne_i64(%a : si64, %b : si64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ne", %a, %b : si64
  return %r : i1
}

func.func @lt_i64(%a : si64, %b : si64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "lt", %a, %b : si64
  return %r : i1
}

func.func @le_i64(%a : si64, %b : si64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "le", %a, %b : si64
  return %r : i1
}

func.func @gt_i64(%a : si64, %b : si64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "gt", %a, %b : si64
  return %r : i1
}

func.func @ge_i64(%a : si64, %b : si64) -> i1 attributes {rust.declare} {
  %r = arc.cmpi "ge", %a, %b : si64
  return %r : i1
}

// ---------------------------------------

func.func @eq_f32(%a : f32, %b : f32) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "oeq", %a, %b : f32
  return %r : i1
}

func.func @ne_f32(%a : f32, %b : f32) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "one", %a, %b : f32
  return %r : i1
}

func.func @lt_f32(%a : f32, %b : f32) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "olt", %a, %b : f32
  return %r : i1
}

func.func @le_f32(%a : f32, %b : f32) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "ole", %a, %b : f32
  return %r : i1
}

func.func @gt_f32(%a : f32, %b : f32) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "ogt", %a, %b : f32
  return %r : i1
}

func.func @ge_f32(%a : f32, %b : f32) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "oge", %a, %b : f32
  return %r : i1
}

func.func @eq_f64(%a : f64, %b : f64) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "oeq", %a, %b : f64
  return %r : i1
}

func.func @ne_f64(%a : f64, %b : f64) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "one", %a, %b : f64
  return %r : i1
}

func.func @lt_f64(%a : f64, %b : f64) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "olt", %a, %b : f64
  return %r : i1
}

func.func @le_f64(%a : f64, %b : f64) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "ole", %a, %b : f64
  return %r : i1
}

func.func @gt_f64(%a : f64, %b : f64) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "ogt", %a, %b : f64
  return %r : i1
}

func.func @ge_f64(%a : f64, %b : f64) -> i1 attributes {rust.declare} {
  %r = arith.cmpf "oge", %a, %b : f64
  return %r : i1
}

// Unary ops

func.func @acos_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = arc.acos %a : f32
  return %r : f32
}

func.func @asin_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = arc.asin %a : f32
  return %r : f32
}

func.func @atan_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = math.atan %a : f32
  return %r : f32
}

func.func @cos_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = math.cos %a : f32
  return %r : f32
}

func.func @cosh_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = arc.cosh %a : f32
  return %r : f32
}

func.func @exp_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = math.exp %a : f32
  return %r : f32
}

func.func @log_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = math.log %a : f32
  return %r : f32
}

func.func @sin_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = math.sin %a : f32
  return %r : f32
}

func.func @sinh_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = arc.sinh %a : f32
  return %r : f32
}

func.func @sqrt_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = math.sqrt %a : f32
  return %r : f32
}

func.func @tan_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = arc.tan %a : f32
  return %r : f32
}

func.func @tanh_f32(%a : f32) -> f32 attributes {rust.declare} {
  %r = math.tanh %a : f32
  return %r : f32
}

func.func @acos_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = arc.acos %a : f64
  return %r : f64
}

func.func @asin_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = arc.asin %a : f64
  return %r : f64
}

func.func @atan_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = math.atan %a : f64
  return %r : f64
}

func.func @cos_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = math.cos %a : f64
  return %r : f64
}

func.func @cosh_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = arc.cosh %a : f64
  return %r : f64
}

func.func @exp_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = math.exp %a : f64
  return %r : f64
}

func.func @log_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = math.log %a : f64
  return %r : f64
}

func.func @sin_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = math.sin %a : f64
  return %r : f64
}

func.func @sinh_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = arc.sinh %a : f64
  return %r : f64
}

func.func @sqrt_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = math.sqrt %a : f64
  return %r : f64
}

func.func @tan_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = arc.tan %a : f64
  return %r : f64
}

func.func @tanh_f64(%a : f64) -> f64 attributes {rust.declare} {
  %r = math.tanh %a : f64
  return %r : f64
}
