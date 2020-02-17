# RUN: arc-to-mlir -i %s | FileCheck %s
# RUN: arc-to-mlir -i %s | arc-mlir | FileCheck %s

let c_bool : bool = true;
let c_i8 : i8 = 127i8;
let c_i16 : i16 = 32767i16;
let c_i32 : i32 = 2147483647;
let c_i64 : i64 = 9223372036854775807i64;
let c_u8 : u8 = 255u8;
let c_u16 : u16 = 65535u16;
let c_u32 : u32 = 4294967295u32;
let c_u64 : u64 = 18446744073709551615u64;
let c_f32 : f32 = 3.4028234664e38f32;
let c_f64 : f64 = 1.7976931348623157e308;

let c1_i8 : i8 = 126i8;
let c1_i16 : i16 = 32766i16;
let c1_i32 : i32 = 2147483646;
let c1_i64 : i64 = 9223372036854775806i64;
let c1_u8 : u8 = 254u8;
let c1_u16 : u16 = 65534u16;
let c1_u32 : u32 = 4294967294u32;
let c1_u64 : u64 = 18446744073709551614u64;
let c1_f32 : f32 = 3.4028234664e37f32;
let c1_f64 : f64 = 1.7976931348623156e308;

let sum_i8 : i8 = c_i8 + c_i8;
#CHECK: {{%[^ ]+}} = addi {{%[^ ]+}}, {{%[^ ]+}} : i8
let sum_i16 : i16 = c_i16 + c_i16;
#CHECK: {{%[^ ]+}} = addi {{%[^ ]+}}, {{%[^ ]+}} : i16
let sum_i32 : i32 = c_i32 + c_i32;
#CHECK: {{%[^ ]+}} = addi {{%[^ ]+}}, {{%[^ ]+}} : i32
let sum_i64 : i64 = c_i64 + c_i64;
#CHECK: {{%[^ ]+}} = addi {{%[^ ]+}}, {{%[^ ]+}} : i64
let sum_u8 : u8 = c_u8 + c_u8;
#CHECK: {{%[^ ]+}} = addi {{%[^ ]+}}, {{%[^ ]+}} : i8
let sum_u16 : u16 = c_u16 + c_u16;
#CHECK: {{%[^ ]+}} = addi {{%[^ ]+}}, {{%[^ ]+}} : i16
let sum_u32 : u32 = c_u32 + c_u32;
#CHECK: {{%[^ ]+}} = addi {{%[^ ]+}}, {{%[^ ]+}} : i32
let sum_u64 : u64 = c_u64 + c_u64;
#CHECK: {{%[^ ]+}} = addi {{%[^ ]+}}, {{%[^ ]+}} : i64
let sum_f32 : f32 = c_f32 + c_f32;
#CHECK: {{%[^ ]+}} = addf {{%[^ ]+}}, {{%[^ ]+}} : f32
let sum_f64 : f64 = c_f64 + c_f64;
#CHECK: {{%[^ ]+}} = addf {{%[^ ]+}}, {{%[^ ]+}} : f64

let difference_i8 : i8 = c_i8 - c_i8;
#CHECK: {{%[^ ]+}} = subi {{%[^ ]+}}, {{%[^ ]+}} : i8
let difference_i16 : i16 = c_i16 - c_i16;
#CHECK: {{%[^ ]+}} = subi {{%[^ ]+}}, {{%[^ ]+}} : i16
let difference_i32 : i32 = c_i32 - c_i32;
#CHECK: {{%[^ ]+}} = subi {{%[^ ]+}}, {{%[^ ]+}} : i32
let difference_i64 : i64 = c_i64 - c_i64;
#CHECK: {{%[^ ]+}} = subi {{%[^ ]+}}, {{%[^ ]+}} : i64
let difference_u8 : u8 = c_u8 - c_u8;
#CHECK: {{%[^ ]+}} = subi {{%[^ ]+}}, {{%[^ ]+}} : i8
let difference_u16 : u16 = c_u16 - c_u16;
#CHECK: {{%[^ ]+}} = subi {{%[^ ]+}}, {{%[^ ]+}} : i16
let difference_u32 : u32 = c_u32 - c_u32;
#CHECK: {{%[^ ]+}} = subi {{%[^ ]+}}, {{%[^ ]+}} : i32
let difference_u64 : u64 = c_u64 - c_u64;
#CHECK: {{%[^ ]+}} = subi {{%[^ ]+}}, {{%[^ ]+}} : i64
let difference_f32 : f32 = c_f32 - c_f32;
#CHECK: {{%[^ ]+}} = subf {{%[^ ]+}}, {{%[^ ]+}} : f32
let difference_f64 : f64 = c_f64 - c_f64;
#CHECK: {{%[^ ]+}} = subf {{%[^ ]+}}, {{%[^ ]+}} : f64

let product_i8 : i8 = c_i8 * c_i8;
#CHECK: {{%[^ ]+}} = muli {{%[^ ]+}}, {{%[^ ]+}} : i8
let product_i16 : i16 = c_i16 * c_i16;
#CHECK: {{%[^ ]+}} = muli {{%[^ ]+}}, {{%[^ ]+}} : i16
let product_i32 : i32 = c_i32 * c_i32;
#CHECK: {{%[^ ]+}} = muli {{%[^ ]+}}, {{%[^ ]+}} : i32
let product_i64 : i64 = c_i64 * c_i64;
#CHECK: {{%[^ ]+}} = muli {{%[^ ]+}}, {{%[^ ]+}} : i64
let product_u8 : u8 = c_u8 * c_u8;
#CHECK: {{%[^ ]+}} = muli {{%[^ ]+}}, {{%[^ ]+}} : i8
let product_u16 : u16 = c_u16 * c_u16;
#CHECK: {{%[^ ]+}} = muli {{%[^ ]+}}, {{%[^ ]+}} : i16
let product_u32 : u32 = c_u32 * c_u32;
#CHECK: {{%[^ ]+}} = muli {{%[^ ]+}}, {{%[^ ]+}} : i32
let product_u64 : u64 = c_u64 * c_u64;
#CHECK: {{%[^ ]+}} = muli {{%[^ ]+}}, {{%[^ ]+}} : i64
let product_f32 : f32 = c_f32 * c_f32;
#CHECK: {{%[^ ]+}} = mulf {{%[^ ]+}}, {{%[^ ]+}} : f32
let product_f64 : f64 = c_f64 * c_f64;
#CHECK: {{%[^ ]+}} = mulf {{%[^ ]+}}, {{%[^ ]+}} : f64

let quotient_i8 : i8 = c_i8 / c_i8;
#CHECK: {{%[^ ]+}} = divi_signed {{%[^ ]+}}, {{%[^ ]+}} : i8
let quotient_i16 : i16 = c_i16 / c_i16;
#CHECK: {{%[^ ]+}} = divi_signed {{%[^ ]+}}, {{%[^ ]+}} : i16
let quotient_i32 : i32 = c_i32 / c_i32;
#CHECK: {{%[^ ]+}} = divi_signed {{%[^ ]+}}, {{%[^ ]+}} : i32
let quotient_i64 : i64 = c_i64 / c_i64;
#CHECK: {{%[^ ]+}} = divi_signed {{%[^ ]+}}, {{%[^ ]+}} : i64
let quotient_u8 : u8 = c_u8 / c_u8;
#CHECK: {{%[^ ]+}} = divi_unsigned {{%[^ ]+}}, {{%[^ ]+}} : i8
let quotient_u16 : u16 = c_u16 / c_u16;
#CHECK: {{%[^ ]+}} = divi_unsigned {{%[^ ]+}}, {{%[^ ]+}} : i16
let quotient_u32 : u32 = c_u32 / c_u32;
#CHECK: {{%[^ ]+}} = divi_unsigned {{%[^ ]+}}, {{%[^ ]+}} : i32
let quotient_u64 : u64 = c_u64 / c_u64;
#CHECK: {{%[^ ]+}} = divi_unsigned {{%[^ ]+}}, {{%[^ ]+}} : i64
let quotient_f32 : f32 = c_f32 / c_f32;
#CHECK: {{%[^ ]+}} = divf {{%[^ ]+}}, {{%[^ ]+}} : f32
let quotient_f64 : f64 = c_f64 / c_f64;
#CHECK: {{%[^ ]+}} = divf {{%[^ ]+}}, {{%[^ ]+}} : f64

let remainder_i8 : i8 = c_i8 % c_i8;
#CHECK: {{%[^ ]+}} = remi_signed {{%[^ ]+}}, {{%[^ ]+}} : i8
let remainder_i16 : i16 = c_i16 % c_i16;
#CHECK: {{%[^ ]+}} = remi_signed {{%[^ ]+}}, {{%[^ ]+}} : i16
let remainder_i32 : i32 = c_i32 % c_i32;
#CHECK: {{%[^ ]+}} = remi_signed {{%[^ ]+}}, {{%[^ ]+}} : i32
let remainder_i64 : i64 = c_i64 % c_i64;
#CHECK: {{%[^ ]+}} = remi_signed {{%[^ ]+}}, {{%[^ ]+}} : i64
let remainder_u8 : u8 = c_u8 % c_u8;
#CHECK: {{%[^ ]+}} = remi_unsigned {{%[^ ]+}}, {{%[^ ]+}} : i8
let remainder_u16 : u16 = c_u16 % c_u16;
#CHECK: {{%[^ ]+}} = remi_unsigned {{%[^ ]+}}, {{%[^ ]+}} : i16
let remainder_u32 : u32 = c_u32 % c_u32;
#CHECK: {{%[^ ]+}} = remi_unsigned {{%[^ ]+}}, {{%[^ ]+}} : i32
let remainder_u64 : u64 = c_u64 % c_u64;
#CHECK: {{%[^ ]+}} = remi_unsigned {{%[^ ]+}}, {{%[^ ]+}} : i64
let remainder_f32 : f32 = c_f32 % c_f32;
#CHECK: {{%[^ ]+}} = remf {{%[^ ]+}}, {{%[^ ]+}} : f32
let remainder_f64 : f64 = c_f64 % c_f64;
#CHECK: {{%[^ ]+}} = remf {{%[^ ]+}}, {{%[^ ]+}} : f64

let lt_i8 : bool = c_i8 < c_i8;
#CHECK: {{%[^ ]+}} = cmpi "slt", {{%[^ ]+}}, {{%[^ ]+}} : i8
let lt_i16 : bool = c_i16 < c_i16;
#CHECK: {{%[^ ]+}} = cmpi "slt", {{%[^ ]+}}, {{%[^ ]+}} : i16
let lt_i32 : bool = c_i32 < c_i32;
#CHECK: {{%[^ ]+}} = cmpi "slt", {{%[^ ]+}}, {{%[^ ]+}} : i32
let lt_i64 : bool = c_i64 < c_i64;
#CHECK: {{%[^ ]+}} = cmpi "slt", {{%[^ ]+}}, {{%[^ ]+}} : i64
let lt_u8 : bool = c_u8 < c_u8;
#CHECK: {{%[^ ]+}} = cmpi "ult", {{%[^ ]+}}, {{%[^ ]+}} : i8
let lt_u16 : bool = c_u16 < c_u16;
#CHECK: {{%[^ ]+}} = cmpi "ult", {{%[^ ]+}}, {{%[^ ]+}} : i16
let lt_u32 : bool = c_u32 < c_u32;
#CHECK: {{%[^ ]+}} = cmpi "ult", {{%[^ ]+}}, {{%[^ ]+}} : i32
let lt_u64 : bool = c_u64 < c_u64;
#CHECK: {{%[^ ]+}} = cmpi "ult", {{%[^ ]+}}, {{%[^ ]+}} : i64
let lt_f32 : bool = c_f32 < c_f32;
#CHECK: {{%[^ ]+}} = cmpf "olt", {{%[^ ]+}}, {{%[^ ]+}} : f32
let lt_f64 : bool = c_f64 < c_f64;
#CHECK: {{%[^ ]+}} = cmpf "olt", {{%[^ ]+}}, {{%[^ ]+}} : f64

let le_i8 : bool = c_i8 <= c_i8;
#CHECK: {{%[^ ]+}} = cmpi "sle", {{%[^ ]+}}, {{%[^ ]+}} : i8
let le_i16 : bool = c_i16 <= c_i16;
#CHECK: {{%[^ ]+}} = cmpi "sle", {{%[^ ]+}}, {{%[^ ]+}} : i16
let le_i32 : bool = c_i32 <= c_i32;
#CHECK: {{%[^ ]+}} = cmpi "sle", {{%[^ ]+}}, {{%[^ ]+}} : i32
let le_i64 : bool = c_i64 <= c_i64;
#CHECK: {{%[^ ]+}} = cmpi "sle", {{%[^ ]+}}, {{%[^ ]+}} : i64
let le_u8 : bool = c_u8 <= c_u8;
#CHECK: {{%[^ ]+}} = cmpi "ule", {{%[^ ]+}}, {{%[^ ]+}} : i8
let le_u16 : bool = c_u16 <= c_u16;
#CHECK: {{%[^ ]+}} = cmpi "ule", {{%[^ ]+}}, {{%[^ ]+}} : i16
let le_u32 : bool = c_u32 <= c_u32;
#CHECK: {{%[^ ]+}} = cmpi "ule", {{%[^ ]+}}, {{%[^ ]+}} : i32
let le_u64 : bool = c_u64 <= c_u64;
#CHECK: {{%[^ ]+}} = cmpi "ule", {{%[^ ]+}}, {{%[^ ]+}} : i64
let le_f32 : bool = c_f32 <= c_f32;
#CHECK: {{%[^ ]+}} = cmpf "ole", {{%[^ ]+}}, {{%[^ ]+}} : f32
let le_f64 : bool = c_f64 <= c_f64;
#CHECK: {{%[^ ]+}} = cmpf "ole", {{%[^ ]+}}, {{%[^ ]+}} : f64

let gt_i8 : bool = c_i8 > c_i8;
#CHECK: {{%[^ ]+}} = cmpi "sgt", {{%[^ ]+}}, {{%[^ ]+}} : i8
let gt_i16 : bool = c_i16 > c_i16;
#CHECK: {{%[^ ]+}} = cmpi "sgt", {{%[^ ]+}}, {{%[^ ]+}} : i16
let gt_i32 : bool = c_i32 > c_i32;
#CHECK: {{%[^ ]+}} = cmpi "sgt", {{%[^ ]+}}, {{%[^ ]+}} : i32
let gt_i64 : bool = c_i64 > c_i64;
#CHECK: {{%[^ ]+}} = cmpi "sgt", {{%[^ ]+}}, {{%[^ ]+}} : i64
let gt_u8 : bool = c_u8 > c_u8;
#CHECK: {{%[^ ]+}} = cmpi "ugt", {{%[^ ]+}}, {{%[^ ]+}} : i8
let gt_u16 : bool = c_u16 > c_u16;
#CHECK: {{%[^ ]+}} = cmpi "ugt", {{%[^ ]+}}, {{%[^ ]+}} : i16
let gt_u32 : bool = c_u32 > c_u32;
#CHECK: {{%[^ ]+}} = cmpi "ugt", {{%[^ ]+}}, {{%[^ ]+}} : i32
let gt_u64 : bool = c_u64 > c_u64;
#CHECK: {{%[^ ]+}} = cmpi "ugt", {{%[^ ]+}}, {{%[^ ]+}} : i64
let gt_f32 : bool = c_f32 > c_f32;
#CHECK: {{%[^ ]+}} = cmpf "ogt", {{%[^ ]+}}, {{%[^ ]+}} : f32
let gt_f64 : bool = c_f64 > c_f64;
#CHECK: {{%[^ ]+}} = cmpf "ogt", {{%[^ ]+}}, {{%[^ ]+}} : f64

let ge_i8 : bool = c_i8 >= c_i8;
#CHECK: {{%[^ ]+}} = cmpi "sge", {{%[^ ]+}}, {{%[^ ]+}} : i8
let ge_i16 : bool = c_i16 >= c_i16;
#CHECK: {{%[^ ]+}} = cmpi "sge", {{%[^ ]+}}, {{%[^ ]+}} : i16
let ge_i32 : bool = c_i32 >= c_i32;
#CHECK: {{%[^ ]+}} = cmpi "sge", {{%[^ ]+}}, {{%[^ ]+}} : i32
let ge_i64 : bool = c_i64 >= c_i64;
#CHECK: {{%[^ ]+}} = cmpi "sge", {{%[^ ]+}}, {{%[^ ]+}} : i64
let ge_u8 : bool = c_u8 >= c_u8;
#CHECK: {{%[^ ]+}} = cmpi "uge", {{%[^ ]+}}, {{%[^ ]+}} : i8
let ge_u16 : bool = c_u16 >= c_u16;
#CHECK: {{%[^ ]+}} = cmpi "uge", {{%[^ ]+}}, {{%[^ ]+}} : i16
let ge_u32 : bool = c_u32 >= c_u32;
#CHECK: {{%[^ ]+}} = cmpi "uge", {{%[^ ]+}}, {{%[^ ]+}} : i32
let ge_u64 : bool = c_u64 >= c_u64;
#CHECK: {{%[^ ]+}} = cmpi "uge", {{%[^ ]+}}, {{%[^ ]+}} : i64
let ge_f32 : bool = c_f32 >= c_f32;
#CHECK: {{%[^ ]+}} = cmpf "oge", {{%[^ ]+}}, {{%[^ ]+}} : f32
let ge_f64 : bool = c_f64 >= c_f64;
#CHECK: {{%[^ ]+}} = cmpf "oge", {{%[^ ]+}}, {{%[^ ]+}} : f64

let eq_bool : bool = c_bool == c_bool;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i1
let eq_i8 : bool = c_i8 == c_i8;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i8
let eq_i16 : bool = c_i16 == c_i16;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i16
let eq_i32 : bool = c_i32 == c_i32;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i32
let eq_i64 : bool = c_i64 == c_i64;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i64
let eq_u8 : bool = c_u8 == c_u8;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i8
let eq_u16 : bool = c_u16 == c_u16;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i16
let eq_u32 : bool = c_u32 == c_u32;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i32
let eq_u64 : bool = c_u64 == c_u64;
#CHECK: {{%[^ ]+}} = cmpi "eq", {{%[^ ]+}}, {{%[^ ]+}} : i64
let eq_f32 : bool = c_f32 == c_f32;
#CHECK: {{%[^ ]+}} = cmpf "oeq", {{%[^ ]+}}, {{%[^ ]+}} : f32
let eq_f64 : bool = c_f64 == c_f64;
#CHECK: {{%[^ ]+}} = cmpf "oeq", {{%[^ ]+}}, {{%[^ ]+}} : f64

let ne_bool : bool = c_bool != c_bool;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i1
let ne_i8 : bool = c_i8 != c_i8;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i8
let ne_i16 : bool = c_i16 != c_i16;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i16
let ne_i32 : bool = c_i32 != c_i32;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i32
let ne_i64 : bool = c_i64 != c_i64;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i64
let ne_u8 : bool = c_u8 != c_u8;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i8
let ne_u16 : bool = c_u16 != c_u16;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i16
let ne_u32 : bool = c_u32 != c_u32;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i32
let ne_u64 : bool = c_u64 != c_u64;
#CHECK: {{%[^ ]+}} = cmpi "ne", {{%[^ ]+}}, {{%[^ ]+}} : i64
let ne_f32 : bool = c_f32 != c_f32;
#CHECK: {{%[^ ]+}} = cmpf "one", {{%[^ ]+}}, {{%[^ ]+}} : f32
let ne_f64 : bool = c_f64 != c_f64;
#CHECK: {{%[^ ]+}} = cmpf "one", {{%[^ ]+}}, {{%[^ ]+}} : f64

let land : bool = c_bool && c_bool;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i1
let lor : bool = c_bool || c_bool;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i1

let band_i8 : i8 = c_i8 & c_i8;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i8
let band_i16 : i16 = c_i16 & c_i16;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i16
let band_i32 : i32 = c_i32 & c_i32;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i32
let band_i64 : i64 = c_i64 & c_i64;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i64
let band_u8 : u8 = c_u8 & c_u8;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i8
let band_u16 : u16 = c_u16 & c_u16;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i16
let band_u32 : u32 = c_u32 & c_u32;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i32
let band_u64 : u64 = c_u64 & c_u64;
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i64

let bor_i8 : i8 = c_i8 | c_i8;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i8
let bor_i16 : i16 = c_i16 | c_i16;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i16
let bor_i32 : i32 = c_i32 | c_i32;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i32
let bor_i64 : i64 = c_i64 | c_i64;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i64
let bor_u8 : u8 = c_u8 | c_u8;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i8
let bor_u16 : u16 = c_u16 | c_u16;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i16
let bor_u32 : u32 = c_u32 | c_u32;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i32
let bor_u64 : u64 = c_u64 | c_u64;
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i64

let bxor_i8 : i8 = c_i8 ^ c_i8;
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i8
let bxor_i16 : i16 = c_i16 ^ c_i16;
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i16
let bxor_i32 : i32 = c_i32 ^ c_i32;
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i32
let bxor_i64 : i64 = c_i64 ^ c_i64;
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i64
let bxor_u8 : u8 = c_u8 ^ c_u8;
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i8
let bxor_u16 : u16 = c_u16 ^ c_u16;
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i16
let bxor_u32 : u32 = c_u32 ^ c_u32;
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i32
let bxor_u64 : u64 = c_u64 ^ c_u64;
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i64

let min_i8 : i8 = min(c_i8, c1_i8);
#CHECK-DAG: [[A0:%[^ ]+]] = cmpi "slt", [[B0:%[^ ]+]], [[C0:%[^ ]+]] : i8
#CHECK: {{%[^ ]+}} = select [[A0]], [[B0]], [[C0]] : i8

let min_i16 : i16 = min(c_i16, c1_i16);
#CHECK-DAG: [[A1:%[^ ]+]] = cmpi "slt", [[B1:%[^ ]+]], [[C1:%[^ ]+]] : i16
#CHECK: {{%[^ ]+}} = select [[A1]], [[B1]], [[C1]] : i16

let min_i32 : i32 = min(c_i32, c1_i32);
#CHECK-DAG: [[A2:%[^ ]+]] = cmpi "slt", [[B2:%[^ ]+]], [[C2:%[^ ]+]] : i32
#CHECK: {{%[^ ]+}} = select [[A2]], [[B2]], [[C2]] : i32

let min_i64 : i64 = min(c_i64, c1_i64);
#CHECK-DAG: [[A3:%[^ ]+]] = cmpi "slt", [[B3:%[^ ]+]], [[C3:%[^ ]+]] : i64
#CHECK: {{%[^ ]+}} = select [[A3]], [[B3]], [[C3]] : i64

let min_u8 : u8 = min(c_u8, c1_u8);
#CHECK-DAG: [[A4:%[^ ]+]] = cmpi "ult", [[B4:%[^ ]+]], [[C4:%[^ ]+]] : i8
#CHECK: {{%[^ ]+}} = select [[A4]], [[B4]], [[C4]] : i8

let min_u16 : u16 = min(c_u16, c1_u16);
#CHECK-DAG: [[A5:%[^ ]+]] = cmpi "ult", [[B5:%[^ ]+]], [[C5:%[^ ]+]] : i16
#CHECK: {{%[^ ]+}} = select [[A5]], [[B5]], [[C5]] : i16

let min_u32 : u32 = min(c_u32, c1_u32);
#CHECK-DAG: [[A6:%[^ ]+]] = cmpi "ult", [[B6:%[^ ]+]], [[C6:%[^ ]+]] : i32
#CHECK: {{%[^ ]+}} = select [[A6]], [[B6]], [[C6]] : i32

let min_u64 : u64 = min(c_u64, c1_u64);
#CHECK-DAG: [[A7:%[^ ]+]] = cmpi "ult", [[B7:%[^ ]+]], [[C7:%[^ ]+]] : i64
#CHECK: {{%[^ ]+}} = select [[A7]], [[B7]], [[C7]] : i64

let min_f32 : f32 = min(c_f32, c1_f32);
#CHECK-DAG: [[A8:%[^ ]+]] = cmpf "olt", [[B8:%[^ ]+]], [[C8:%[^ ]+]] : f32
#CHECK: {{%[^ ]+}} = select [[A8]], [[B8]], [[C8]] : f32

let min_f64 : f64 = min(c_f64, c1_f64);
#CHECK-DAG: [[A9:%[^ ]+]] = cmpf "olt", [[B9:%[^ ]+]], [[C9:%[^ ]+]] : f64
#CHECK: {{%[^ ]+}} = select [[A9]], [[B9]], [[C9]] : f64

let max_i8 : i8 = max(c_i8, c1_i8);
#CHECK-DAG: [[A0:%[^ ]+]] = cmpi "slt", [[B0:%[^ ]+]], [[C0:%[^ ]+]] : i8
#CHECK: {{%[^ ]+}} = select [[A0]], [[C0]], [[B0]] : i8

let max_i16 : i16 = max(c_i16, c1_i16);
#CHECK-DAG: [[A1:%[^ ]+]] = cmpi "slt", [[B1:%[^ ]+]], [[C1:%[^ ]+]] : i16
#CHECK: {{%[^ ]+}} = select [[A1]], [[C1]], [[B1]] : i16

let max_i32 : i32 = max(c_i32, c1_i32);
#CHECK-DAG: [[A2:%[^ ]+]] = cmpi "slt", [[B2:%[^ ]+]], [[C2:%[^ ]+]] : i32
#CHECK: {{%[^ ]+}} = select [[A2]], [[C2]], [[B2]] : i32

let max_i64 : i64 = max(c_i64, c1_i64);
#CHECK-DAG: [[A3:%[^ ]+]] = cmpi "slt", [[B3:%[^ ]+]], [[C3:%[^ ]+]] : i64
#CHECK: {{%[^ ]+}} = select [[A3]], [[C3]], [[B3]] : i64

let max_u8 : u8 = max(c_u8, c1_u8);
#CHECK-DAG: [[A4:%[^ ]+]] = cmpi "ult", [[B4:%[^ ]+]], [[C4:%[^ ]+]] : i8
#CHECK: {{%[^ ]+}} = select [[A4]], [[C4]], [[B4]] : i8

let max_u16 : u16 = max(c_u16, c1_u16);
#CHECK-DAG: [[A5:%[^ ]+]] = cmpi "ult", [[B5:%[^ ]+]], [[C5:%[^ ]+]] : i16
#CHECK: {{%[^ ]+}} = select [[A5]], [[C5]], [[B5]] : i16

let max_u32 : u32 = max(c_u32, c1_u32);
#CHECK-DAG: [[A6:%[^ ]+]] = cmpi "ult", [[B6:%[^ ]+]], [[C6:%[^ ]+]] : i32
#CHECK: {{%[^ ]+}} = select [[A6]], [[C6]], [[B6]] : i32

let max_u64 : u64 = max(c_u64, c1_u64);
#CHECK-DAG: [[A7:%[^ ]+]] = cmpi "ult", [[B7:%[^ ]+]], [[C7:%[^ ]+]] : i64
#CHECK: {{%[^ ]+}} = select [[A7]], [[C7]], [[B7]] : i64

let max_f32 : f32 = max(c_f32, c1_f32);
#CHECK-DAG: [[A8:%[^ ]+]] = cmpf "olt", [[B8:%[^ ]+]], [[C8:%[^ ]+]] : f32
#CHECK: {{%[^ ]+}} = select [[A8]], [[C8]], [[B8]] : f32

let max_f64 : f64 = max(c_f64, c1_f64);
#CHECK-DAG: [[A9:%[^ ]+]] = cmpf "olt", [[B9:%[^ ]+]], [[C9:%[^ ]+]] : f64
#CHECK: {{%[^ ]+}} = select [[A9]], [[C9]], [[B9]] : f64

let pow_f32: f32 = pow(c_f32, c1_f32);
#CHECK-DAG: [[AA0:%[^ ]+]] = log [[AA1:%[^ ]+]] : f32
#CHECK-DAG: [[AA2:%[^ ]+]] = mulf [[AA0:%[^ ]+]], [[AA3:%[^ ]+]] : f32
#CHECK: {{%[^ ]+}} = exp [[AA2]] : f32

let pow_f64: f64 = pow(c_f64, c1_f64);
#CHECK-DAG: [[AA0:%[^ ]+]] = log [[AA1:%[^ ]+]] : f64
#CHECK-DAG: [[AA2:%[^ ]+]] = mulf [[AA0:%[^ ]+]], [[AA3:%[^ ]+]] : f64
#CHECK: {{%[^ ]+}} = exp [[AA2]] : f64

4711
