# RUN: arc-script run --output=MLIR %s | FileCheck %s
# RUN: arc-script run --output=MLIR %s | arc-mlir | FileCheck %s
# RUN: arc-script run --output=MLIR %s | arc-mlir -arc-to-rust

fun main() {

    let c_bool: bool = true in
    let c_i8: i8 = 127i8 in
    let c_i16: i16 = 32767i16 in
    let c_i32: i32 = 2147483647 in
    let c_i64: i64 = 9223372036854775807i64 in
    let c_u8: u8 = 255u8 in
    let c_u16: u16 = 65535u16 in
    let c_u32: u32 = 4294967295u32 in
    let c_u64: u64 = 18446744073709551615u64 in
    let c_f32: f32 = 3.4028234664e38f32 in
    let c_f64: f64 = 1.7976931348623157e308 in

    let c1_i8: i8 = 126i8 in
    let c1_i16: i16 = 32766i16 in
    let c1_i32: i32 = 2147483646 in
    let c1_i64: i64 = 9223372036854775806i64 in
    let c1_u8: u8 = 254u8 in
    let c1_u16: u16 = 65534u16 in
    let c1_u32: u32 = 4294967294u32 in
    let c1_u64: u64 = 18446744073709551614u64 in
    let c1_f32: f32 = 3.4028234664e37f32 in
    let c1_f64: f64 = 1.7976931348623156e308 in

    let sum_i8: i8 = c_i8 + c_i8 in
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : si8
    let sum_i16: i16 = c_i16 + c_i16 in
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : si16
    let sum_i32: i32 = c_i32 + c_i32 in
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : si32
    let sum_i64: i64 = c_i64 + c_i64 in
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : si64
    let sum_u8: u8 = c_u8 + c_u8 in
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let sum_u16: u16 = c_u16 + c_u16 in
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let sum_u32: u32 = c_u32 + c_u32 in
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let sum_u64: u64 = c_u64 + c_u64 in
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let sum_f32: f32 = c_f32 + c_f32 in
#CHECK: {{%[^ ]+}} = addf {{%[^ ]+}}, {{%[^ ]+}} : f32
    let sum_f64: f64 = c_f64 + c_f64 in
#CHECK: {{%[^ ]+}} = addf {{%[^ ]+}}, {{%[^ ]+}} : f64

    let difference_i8: i8 = c_i8 - c_i8 in
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : si8
    let difference_i16: i16 = c_i16 - c_i16 in
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : si16
    let difference_i32: i32 = c_i32 - c_i32 in
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : si32
    let difference_i64: i64 = c_i64 - c_i64 in
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : si64
    let difference_u8: u8 = c_u8 - c_u8 in
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let difference_u16: u16 = c_u16 - c_u16 in
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let difference_u32: u32 = c_u32 - c_u32 in
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let difference_u64: u64 = c_u64 - c_u64 in
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let difference_f32: f32 = c_f32 - c_f32 in
#CHECK: {{%[^ ]+}} = subf {{%[^ ]+}}, {{%[^ ]+}} : f32
    let difference_f64: f64 = c_f64 - c_f64 in
#CHECK: {{%[^ ]+}} = subf {{%[^ ]+}}, {{%[^ ]+}} : f64

    let product_i8: i8 = c_i8 * c_i8 in
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : si8
    let product_i16: i16 = c_i16 * c_i16 in
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : si16
    let product_i32: i32 = c_i32 * c_i32 in
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : si32
    let product_i64: i64 = c_i64 * c_i64 in
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : si64
    let product_u8: u8 = c_u8 * c_u8 in
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let product_u16: u16 = c_u16 * c_u16 in
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let product_u32: u32 = c_u32 * c_u32 in
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let product_u64: u64 = c_u64 * c_u64 in
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let product_f32: f32 = c_f32 * c_f32 in
#CHECK: {{%[^ ]+}} = mulf {{%[^ ]+}}, {{%[^ ]+}} : f32
    let product_f64: f64 = c_f64 * c_f64 in
#CHECK: {{%[^ ]+}} = mulf {{%[^ ]+}}, {{%[^ ]+}} : f64

    let quotient_i8: i8 = c_i8 / c_i8 in
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : si8
    let quotient_i16: i16 = c_i16 / c_i16 in
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : si16
    let quotient_i32: i32 = c_i32 / c_i32 in
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : si32
    let quotient_i64: i64 = c_i64 / c_i64 in
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : si64
    let quotient_u8: u8 = c_u8 / c_u8 in
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let quotient_u16: u16 = c_u16 / c_u16 in
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let quotient_u32: u32 = c_u32 / c_u32 in
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let quotient_u64: u64 = c_u64 / c_u64 in
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let quotient_f32: f32 = c_f32 / c_f32 in
#CHECK: {{%[^ ]+}} = divf {{%[^ ]+}}, {{%[^ ]+}} : f32
    let quotient_f64: f64 = c_f64 / c_f64 in
#CHECK: {{%[^ ]+}} = divf {{%[^ ]+}}, {{%[^ ]+}} : f64

    let remainder_i8: i8 = c_i8 % c_i8 in
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : si8
    let remainder_i16: i16 = c_i16 % c_i16 in
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : si16
    let remainder_i32: i32 = c_i32 % c_i32 in
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : si32
    let remainder_i64: i64 = c_i64 % c_i64 in
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : si64
    let remainder_u8: u8 = c_u8 % c_u8 in
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let remainder_u16: u16 = c_u16 % c_u16 in
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let remainder_u32: u32 = c_u32 % c_u32 in
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let remainder_u64: u64 = c_u64 % c_u64 in
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let remainder_f32: f32 = c_f32 % c_f32 in
#CHECK: {{%[^ ]+}} = remf {{%[^ ]+}}, {{%[^ ]+}} : f32
    let remainder_f64: f64 = c_f64 % c_f64 in
#CHECK: {{%[^ ]+}} = remf {{%[^ ]+}}, {{%[^ ]+}} : f64

    let lt_i8: bool = c_i8 < c_i8 in
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : si8
    let lt_i16: bool = c_i16 < c_i16 in
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : si16
    let lt_i32: bool = c_i32 < c_i32 in
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : si32
    let lt_i64: bool = c_i64 < c_i64 in
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : si64
    let lt_u8: bool = c_u8 < c_u8 in
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let lt_u16: bool = c_u16 < c_u16 in
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let lt_u32: bool = c_u32 < c_u32 in
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let lt_u64: bool = c_u64 < c_u64 in
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let lt_f32: bool = c_f32 < c_f32 in
#CHECK: {{%[^ ]+}} = cmpf olt, {{%[^ ]+}}, {{%[^ ]+}} : f32
    let lt_f64: bool = c_f64 < c_f64 in
#CHECK: {{%[^ ]+}} = cmpf olt, {{%[^ ]+}}, {{%[^ ]+}} : f64

    let le_i8: bool = c_i8 <= c_i8 in
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : si8
    let le_i16: bool = c_i16 <= c_i16 in
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : si16
    let le_i32: bool = c_i32 <= c_i32 in
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : si32
    let le_i64: bool = c_i64 <= c_i64 in
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : si64
    let le_u8: bool = c_u8 <= c_u8 in
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let le_u16: bool = c_u16 <= c_u16 in
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let le_u32: bool = c_u32 <= c_u32 in
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let le_u64: bool = c_u64 <= c_u64 in
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let le_f32: bool = c_f32 <= c_f32 in
#CHECK: {{%[^ ]+}} = cmpf ole, {{%[^ ]+}}, {{%[^ ]+}} : f32
    let le_f64: bool = c_f64 <= c_f64 in
#CHECK: {{%[^ ]+}} = cmpf ole, {{%[^ ]+}}, {{%[^ ]+}} : f64

    let gt_i8: bool = c_i8 > c_i8 in
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : si8
    let gt_i16: bool = c_i16 > c_i16 in
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : si16
    let gt_i32: bool = c_i32 > c_i32 in
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : si32
    let gt_i64: bool = c_i64 > c_i64 in
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : si64
    let gt_u8: bool = c_u8 > c_u8 in
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let gt_u16: bool = c_u16 > c_u16 in
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let gt_u32: bool = c_u32 > c_u32 in
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let gt_u64: bool = c_u64 > c_u64 in
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let gt_f32: bool = c_f32 > c_f32 in
#CHECK: {{%[^ ]+}} = cmpf ogt, {{%[^ ]+}}, {{%[^ ]+}} : f32
    let gt_f64: bool = c_f64 > c_f64 in
#CHECK: {{%[^ ]+}} = cmpf ogt, {{%[^ ]+}}, {{%[^ ]+}} : f64

    let ge_i8: bool = c_i8 >= c_i8 in
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : si8
    let ge_i16: bool = c_i16 >= c_i16 in
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : si16
    let ge_i32: bool = c_i32 >= c_i32 in
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : si32
    let ge_i64: bool = c_i64 >= c_i64 in
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : si64
    let ge_u8: bool = c_u8 >= c_u8 in
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let ge_u16: bool = c_u16 >= c_u16 in
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let ge_u32: bool = c_u32 >= c_u32 in
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let ge_u64: bool = c_u64 >= c_u64 in
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let ge_f32: bool = c_f32 >= c_f32 in
#CHECK: {{%[^ ]+}} = cmpf oge, {{%[^ ]+}}, {{%[^ ]+}} : f32
    let ge_f64: bool = c_f64 >= c_f64 in
#CHECK: {{%[^ ]+}} = cmpf oge, {{%[^ ]+}}, {{%[^ ]+}} : f64

    let eq_bool: bool = c_bool == c_bool in
#CHECK: {{%[^ ]+}} = cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : i1
    let eq_i8: bool = c_i8 == c_i8 in
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : si8
    let eq_i16: bool = c_i16 == c_i16 in
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : si16
    let eq_i32: bool = c_i32 == c_i32 in
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : si32
    let eq_i64: bool = c_i64 == c_i64 in
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : si64
    let eq_u8: bool = c_u8 == c_u8 in
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let eq_u16: bool = c_u16 == c_u16 in
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let eq_u32: bool = c_u32 == c_u32 in
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let eq_u64: bool = c_u64 == c_u64 in
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let eq_f32: bool = c_f32 == c_f32 in
#CHECK: {{%[^ ]+}} = cmpf oeq, {{%[^ ]+}}, {{%[^ ]+}} : f32
    let eq_f64: bool = c_f64 == c_f64 in
#CHECK: {{%[^ ]+}} = cmpf oeq, {{%[^ ]+}}, {{%[^ ]+}} : f64

    let ne_bool: bool = c_bool != c_bool in
#CHECK: {{%[^ ]+}} = cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : i1
    let ne_i8: bool = c_i8 != c_i8 in
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : si8
    let ne_i16: bool = c_i16 != c_i16 in
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : si16
    let ne_i32: bool = c_i32 != c_i32 in
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : si32
    let ne_i64: bool = c_i64 != c_i64 in
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : si64
    let ne_u8: bool = c_u8 != c_u8 in
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let ne_u16: bool = c_u16 != c_u16 in
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let ne_u32: bool = c_u32 != c_u32 in
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let ne_u64: bool = c_u64 != c_u64 in
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    let ne_f32: bool = c_f32 != c_f32 in
#CHECK: {{%[^ ]+}} = cmpf one, {{%[^ ]+}}, {{%[^ ]+}} : f32
    let ne_f64: bool = c_f64 != c_f64 in
#CHECK: {{%[^ ]+}} = cmpf one, {{%[^ ]+}}, {{%[^ ]+}} : f64

    let land: bool = c_bool and c_bool in
#CHECK: {{%[^ ]+}} = and {{%[^ ]+}}, {{%[^ ]+}} : i1
    let lor: bool = c_bool or c_bool in
#CHECK: {{%[^ ]+}} = or {{%[^ ]+}}, {{%[^ ]+}} : i1
    let lxor: bool = c_bool xor c_bool in
#CHECK: {{%[^ ]+}} = xor {{%[^ ]+}}, {{%[^ ]+}} : i1

    let band_i8: i8 = c_i8 band c_i8 in
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : si8
    let band_i16: i16 = c_i16 band c_i16 in
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : si16
    let band_i32: i32 = c_i32 band c_i32 in
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : si32
    let band_i64: i64 = c_i64 band c_i64 in
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : si64
    let band_u8: u8 = c_u8 band c_u8 in
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let band_u16: u16 = c_u16 band c_u16 in
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let band_u32: u32 = c_u32 band c_u32 in
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let band_u64: u64 = c_u64 band c_u64 in
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : ui64

    let bor_i8: i8 = c_i8 bor c_i8 in
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : si8
    let bor_i16: i16 = c_i16 bor c_i16 in
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : si16
    let bor_i32: i32 = c_i32 bor c_i32 in
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : si32
    let bor_i64: i64 = c_i64 bor c_i64 in
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : si64
    let bor_u8: u8 = c_u8 bor c_u8 in
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let bor_u16: u16 = c_u16 bor c_u16 in
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let bor_u32: u32 = c_u32 bor c_u32 in
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let bor_u64: u64 = c_u64 bor c_u64 in
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : ui64

    let bxor_i8: i8 = c_i8 bxor c_i8 in
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : si8
    let bxor_i16: i16 = c_i16 bxor c_i16 in
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : si16
    let bxor_i32: i32 = c_i32 bxor c_i32 in
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : si32
    let bxor_i64: i64 = c_i64 bxor c_i64 in
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : si64
    let bxor_u8: u8 = c_u8 bxor c_u8 in
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : ui8
    let bxor_u16: u16 = c_u16 bxor c_u16 in
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : ui16
    let bxor_u32: u32 = c_u32 bxor c_u32 in
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : ui32
    let bxor_u64: u64 = c_u64 bxor c_u64 in
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : ui64

##let min_i8: i8 = min(c_i8, c1_i8) in
###XCHECK-DAG: [[A0:%[^ ]+]] = arc.cmpi lt, [[B0:%[^ ]+]], [[C0:%[^ ]+]] : si8
###XCHECK: {{%[^ ]+}} = arc.select [[A0]], [[B0]], [[C0]] : si8
##
##let min_i16: i16 = min(c_i16, c1_i16) in
###XCHECK-DAG: [[A1:%[^ ]+]] = arc.cmpi lt, [[B1:%[^ ]+]], [[C1:%[^ ]+]] : si16
###XCHECK: {{%[^ ]+}} = arc.select [[A1]], [[B1]], [[C1]] : si16
##
##let min_i32: i32 = min(c_i32, c1_i32) in
###XCHECK-DAG: [[A2:%[^ ]+]] = arc.cmpi lt, [[B2:%[^ ]+]], [[C2:%[^ ]+]] : si32
###XCHECK: {{%[^ ]+}} = arc.select [[A2]], [[B2]], [[C2]] : si32
##
##let min_i64: i64 = min(c_i64, c1_i64) in
###XCHECK-DAG: [[A3:%[^ ]+]] = arc.cmpi lt, [[B3:%[^ ]+]], [[C3:%[^ ]+]] : si64
###XCHECK: {{%[^ ]+}} = arc.select [[A3]], [[B3]], [[C3]] : si64
##
##let min_u8: u8 = min(c_u8, c1_u8) in
###XCHECK-DAG: [[A4:%[^ ]+]] = arc.cmpi lt, [[B4:%[^ ]+]], [[C4:%[^ ]+]] : ui8
###XCHECK: {{%[^ ]+}} = arc.select [[A4]], [[B4]], [[C4]] : ui8
##
##let min_u16: u16 = min(c_u16, c1_u16) in
###XCHECK-DAG: [[A5:%[^ ]+]] = arc.cmpi lt, [[B5:%[^ ]+]], [[C5:%[^ ]+]] : ui16
###XCHECK: {{%[^ ]+}} = arc.select [[A5]], [[B5]], [[C5]] : ui16
##
##let min_u32: u32 = min(c_u32, c1_u32) in
###XCHECK-DAG: [[A6:%[^ ]+]] = arc.cmpi lt, [[B6:%[^ ]+]], [[C6:%[^ ]+]] : ui32
###XCHECK: {{%[^ ]+}} = arc.select [[A6]], [[B6]], [[C6]] : ui32
##
##let min_u64: u64 = min(c_u64, c1_u64) in
###XCHECK-DAG: [[A7:%[^ ]+]] = arc.cmpi lt, [[B7:%[^ ]+]], [[C7:%[^ ]+]] : ui64
###XCHECK: {{%[^ ]+}} = arc.select [[A7]], [[B7]], [[C7]] : ui64
##
##let min_f32: f32 = min(c_f32, c1_f32) in
###XCHECK-DAG: [[A8:%[^ ]+]] = cmpf olt, [[B8:%[^ ]+]], [[C8:%[^ ]+]] : f32
###XCHECK: {{%[^ ]+}} = select [[A8]], [[B8]], [[C8]] : f32
##
##let min_f64: f64 = min(c_f64, c1_f64) in
###XCHECK-DAG: [[A9:%[^ ]+]] = cmpf olt, [[B9:%[^ ]+]], [[C9:%[^ ]+]] : f64
###XCHECK: {{%[^ ]+}} = select [[A9]], [[B9]], [[C9]] : f64
##
##let max_i8: i8 = max(c_i8, c1_i8) in
###XCHECK-DAG: [[A0:%[^ ]+]] = arc.cmpi lt, [[B0:%[^ ]+]], [[C0:%[^ ]+]] : si8
###XCHECK: {{%[^ ]+}} = arc.select [[A0]], [[C0]], [[B0]] : si8
##
##let max_i16: i16 = max(c_i16, c1_i16) in
###XCHECK-DAG: [[A1:%[^ ]+]] = arc.cmpi lt, [[B1:%[^ ]+]], [[C1:%[^ ]+]] : si16
###XCHECK: {{%[^ ]+}} = arc.select [[A1]], [[C1]], [[B1]] : si16
##
##let max_i32: i32 = max(c_i32, c1_i32) in
###XCHECK-DAG: [[A2:%[^ ]+]] = arc.cmpi lt, [[B2:%[^ ]+]], [[C2:%[^ ]+]] : si32
###XCHECK: {{%[^ ]+}} = arc.select [[A2]], [[C2]], [[B2]] : si32
##
##let max_i64: i64 = max(c_i64, c1_i64) in
###XCHECK-DAG: [[A3:%[^ ]+]] = arc.cmpi lt, [[B3:%[^ ]+]], [[C3:%[^ ]+]] : si64
###XCHECK: {{%[^ ]+}} = arc.select [[A3]], [[C3]], [[B3]] : si64
##
##let max_u8: u8 = max(c_u8, c1_u8) in
###XCHECK-DAG: [[A4:%[^ ]+]] = arc.cmpi lt, [[B4:%[^ ]+]], [[C4:%[^ ]+]] : ui8
###XCHECK: {{%[^ ]+}} = arc.select [[A4]], [[C4]], [[B4]] : ui8
##
##let max_u16: u16 = max(c_u16, c1_u16) in
###XCHECK-DAG: [[A5:%[^ ]+]] = arc.cmpi lt, [[B5:%[^ ]+]], [[C5:%[^ ]+]] : ui16
###XCHECK: {{%[^ ]+}} = arc.select [[A5]], [[C5]], [[B5]] : ui16
##
##let max_u32: u32 = max(c_u32, c1_u32) in
###XCHECK-DAG: [[A6:%[^ ]+]] = arc.cmpi lt, [[B6:%[^ ]+]], [[C6:%[^ ]+]] : ui32
###XCHECK: {{%[^ ]+}} = arc.select [[A6]], [[C6]], [[B6]] : ui32
##
##let max_u64: u64 = max(c_u64, c1_u64) in
###XCHECK-DAG: [[A7:%[^ ]+]] = arc.cmpi lt, [[B7:%[^ ]+]], [[C7:%[^ ]+]] : ui64
###XCHECK: {{%[^ ]+}} = arc.select [[A7]], [[C7]], [[B7]] : ui64
##
##let max_f32: f32 = max(c_f32, c1_f32) in
###XCHECK-DAG: [[A8:%[^ ]+]] = cmpf olt, [[B8:%[^ ]+]], [[C8:%[^ ]+]] : f32
###XCHECK: {{%[^ ]+}} = select [[A8]], [[C8]], [[B8]] : f32
##
##let max_f64: f64 = max(c_f64, c1_f64) in
###XCHECK-DAG: [[A9:%[^ ]+]] = cmpf olt, [[B9:%[^ ]+]], [[C9:%[^ ]+]] : f64
###XCHECK: {{%[^ ]+}} = select [[A9]], [[C9]], [[B9]] : f64

    let pow_f32: f32 = c_f32 ** c1_f32 in
#CHECK: {{%[^ ]+}} = math.powf {{%[^ ]+}}, {{%[^ ]+}} : f32

    let pow_f64: f64 = c_f64 ** c1_f64 in
#CHECK: {{%[^ ]+}} = math.powf {{%[^ ]+}}, {{%[^ ]+}} : f64

    unit
}
