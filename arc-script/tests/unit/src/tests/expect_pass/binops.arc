# RUN: arc-script --no-prelude run --output=MLIR %s | FileCheck %s
# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir | FileCheck %s
# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir -arc-to-rust

fun main() {

    val c_bool: bool = true;
    val c_i8: i8 = 127i8;
    val c_i16: i16 = 32767i16;
    val c_i32: i32 = 2147483647;
    val c_i64: i64 = 9223372036854775807i64;
    val c_u8: u8 = 255u8;
    val c_u16: u16 = 65535u16;
    val c_u32: u32 = 4294967295u32;
    val c_u64: u64 = 18446744073709551615u64;
    val c_f32: f32 = 3.4028234664e38f32;
    val c_f64: f64 = 1.7976931348623157e308;

    val c1_i8: i8 = 126i8;
    val c1_i16: i16 = 32766i16;
    val c1_i32: i32 = 2147483646;
    val c1_i64: i64 = 9223372036854775806i64;
    val c1_u8: u8 = 254u8;
    val c1_u16: u16 = 65534u16;
    val c1_u32: u32 = 4294967294u32;
    val c1_u64: u64 = 18446744073709551614u64;
    val c1_f32: f32 = 3.4028234664e37f32;
    val c1_f64: f64 = 1.7976931348623156e308;

    val sum_i8: i8 = c_i8 + c_i8;
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : si8
    val sum_i16: i16 = c_i16 + c_i16;
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : si16
    val sum_i32: i32 = c_i32 + c_i32;
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : si32
    val sum_i64: i64 = c_i64 + c_i64;
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : si64
    val sum_u8: u8 = c_u8 + c_u8;
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val sum_u16: u16 = c_u16 + c_u16;
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val sum_u32: u32 = c_u32 + c_u32;
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val sum_u64: u64 = c_u64 + c_u64;
#CHECK: {{%[^ ]+}} = arc.addi {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val sum_f32: f32 = c_f32 + c_f32;
#CHECK: {{%[^ ]+}} = arith.addf {{%[^ ]+}}, {{%[^ ]+}} : f32
    val sum_f64: f64 = c_f64 + c_f64;
#CHECK: {{%[^ ]+}} = arith.addf {{%[^ ]+}}, {{%[^ ]+}} : f64

    val difference_i8: i8 = c_i8 - c_i8;
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : si8
    val difference_i16: i16 = c_i16 - c_i16;
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : si16
    val difference_i32: i32 = c_i32 - c_i32;
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : si32
    val difference_i64: i64 = c_i64 - c_i64;
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : si64
    val difference_u8: u8 = c_u8 - c_u8;
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val difference_u16: u16 = c_u16 - c_u16;
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val difference_u32: u32 = c_u32 - c_u32;
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val difference_u64: u64 = c_u64 - c_u64;
#CHECK: {{%[^ ]+}} = arc.subi {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val difference_f32: f32 = c_f32 - c_f32;
#CHECK: {{%[^ ]+}} = arith.subf {{%[^ ]+}}, {{%[^ ]+}} : f32
    val difference_f64: f64 = c_f64 - c_f64;
#CHECK: {{%[^ ]+}} = arith.subf {{%[^ ]+}}, {{%[^ ]+}} : f64

    val product_i8: i8 = c_i8 * c_i8;
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : si8
    val product_i16: i16 = c_i16 * c_i16;
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : si16
    val product_i32: i32 = c_i32 * c_i32;
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : si32
    val product_i64: i64 = c_i64 * c_i64;
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : si64
    val product_u8: u8 = c_u8 * c_u8;
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val product_u16: u16 = c_u16 * c_u16;
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val product_u32: u32 = c_u32 * c_u32;
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val product_u64: u64 = c_u64 * c_u64;
#CHECK: {{%[^ ]+}} = arc.muli {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val product_f32: f32 = c_f32 * c_f32;
#CHECK: {{%[^ ]+}} = arith.mulf {{%[^ ]+}}, {{%[^ ]+}} : f32
    val product_f64: f64 = c_f64 * c_f64;
#CHECK: {{%[^ ]+}} = arith.mulf {{%[^ ]+}}, {{%[^ ]+}} : f64

    val quotient_i8: i8 = c_i8 / c_i8;
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : si8
    val quotient_i16: i16 = c_i16 / c_i16;
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : si16
    val quotient_i32: i32 = c_i32 / c_i32;
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : si32
    val quotient_i64: i64 = c_i64 / c_i64;
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : si64
    val quotient_u8: u8 = c_u8 / c_u8;
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val quotient_u16: u16 = c_u16 / c_u16;
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val quotient_u32: u32 = c_u32 / c_u32;
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val quotient_u64: u64 = c_u64 / c_u64;
#CHECK: {{%[^ ]+}} = arc.divi {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val quotient_f32: f32 = c_f32 / c_f32;
#CHECK: {{%[^ ]+}} = arith.divf {{%[^ ]+}}, {{%[^ ]+}} : f32
    val quotient_f64: f64 = c_f64 / c_f64;
#CHECK: {{%[^ ]+}} = arith.divf {{%[^ ]+}}, {{%[^ ]+}} : f64

    val remainder_i8: i8 = c_i8 % c_i8;
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : si8
    val remainder_i16: i16 = c_i16 % c_i16;
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : si16
    val remainder_i32: i32 = c_i32 % c_i32;
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : si32
    val remainder_i64: i64 = c_i64 % c_i64;
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : si64
    val remainder_u8: u8 = c_u8 % c_u8;
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val remainder_u16: u16 = c_u16 % c_u16;
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val remainder_u32: u32 = c_u32 % c_u32;
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val remainder_u64: u64 = c_u64 % c_u64;
#CHECK: {{%[^ ]+}} = arc.remi {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val remainder_f32: f32 = c_f32 % c_f32;
#CHECK: {{%[^ ]+}} = arith.remf {{%[^ ]+}}, {{%[^ ]+}} : f32
    val remainder_f64: f64 = c_f64 % c_f64;
#CHECK: {{%[^ ]+}} = arith.remf {{%[^ ]+}}, {{%[^ ]+}} : f64

    val lt_i8: bool = c_i8 < c_i8;
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : si8
    val lt_i16: bool = c_i16 < c_i16;
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : si16
    val lt_i32: bool = c_i32 < c_i32;
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : si32
    val lt_i64: bool = c_i64 < c_i64;
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : si64
    val lt_u8: bool = c_u8 < c_u8;
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val lt_u16: bool = c_u16 < c_u16;
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val lt_u32: bool = c_u32 < c_u32;
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val lt_u64: bool = c_u64 < c_u64;
#CHECK: {{%[^ ]+}} = arc.cmpi lt, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val lt_f32: bool = c_f32 < c_f32;
#CHECK: {{%[^ ]+}} = arith.cmpf olt, {{%[^ ]+}}, {{%[^ ]+}} : f32
    val lt_f64: bool = c_f64 < c_f64;
#CHECK: {{%[^ ]+}} = arith.cmpf olt, {{%[^ ]+}}, {{%[^ ]+}} : f64

    val le_i8: bool = c_i8 <= c_i8;
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : si8
    val le_i16: bool = c_i16 <= c_i16;
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : si16
    val le_i32: bool = c_i32 <= c_i32;
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : si32
    val le_i64: bool = c_i64 <= c_i64;
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : si64
    val le_u8: bool = c_u8 <= c_u8;
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val le_u16: bool = c_u16 <= c_u16;
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val le_u32: bool = c_u32 <= c_u32;
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val le_u64: bool = c_u64 <= c_u64;
#CHECK: {{%[^ ]+}} = arc.cmpi le, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val le_f32: bool = c_f32 <= c_f32;
#CHECK: {{%[^ ]+}} = arith.cmpf ole, {{%[^ ]+}}, {{%[^ ]+}} : f32
    val le_f64: bool = c_f64 <= c_f64;
#CHECK: {{%[^ ]+}} = arith.cmpf ole, {{%[^ ]+}}, {{%[^ ]+}} : f64

    val gt_i8: bool = c_i8 > c_i8;
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : si8
    val gt_i16: bool = c_i16 > c_i16;
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : si16
    val gt_i32: bool = c_i32 > c_i32;
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : si32
    val gt_i64: bool = c_i64 > c_i64;
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : si64
    val gt_u8: bool = c_u8 > c_u8;
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val gt_u16: bool = c_u16 > c_u16;
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val gt_u32: bool = c_u32 > c_u32;
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val gt_u64: bool = c_u64 > c_u64;
#CHECK: {{%[^ ]+}} = arc.cmpi gt, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val gt_f32: bool = c_f32 > c_f32;
#CHECK: {{%[^ ]+}} = arith.cmpf ogt, {{%[^ ]+}}, {{%[^ ]+}} : f32
    val gt_f64: bool = c_f64 > c_f64;
#CHECK: {{%[^ ]+}} = arith.cmpf ogt, {{%[^ ]+}}, {{%[^ ]+}} : f64

    val ge_i8: bool = c_i8 >= c_i8;
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : si8
    val ge_i16: bool = c_i16 >= c_i16;
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : si16
    val ge_i32: bool = c_i32 >= c_i32;
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : si32
    val ge_i64: bool = c_i64 >= c_i64;
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : si64
    val ge_u8: bool = c_u8 >= c_u8;
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val ge_u16: bool = c_u16 >= c_u16;
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val ge_u32: bool = c_u32 >= c_u32;
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val ge_u64: bool = c_u64 >= c_u64;
#CHECK: {{%[^ ]+}} = arc.cmpi ge, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val ge_f32: bool = c_f32 >= c_f32;
#CHECK: {{%[^ ]+}} = arith.cmpf oge, {{%[^ ]+}}, {{%[^ ]+}} : f32
    val ge_f64: bool = c_f64 >= c_f64;
#CHECK: {{%[^ ]+}} = arith.cmpf oge, {{%[^ ]+}}, {{%[^ ]+}} : f64

    val eq_bool: bool = c_bool == c_bool;
#CHECK: {{%[^ ]+}} = arith.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : i1
    val eq_i8: bool = c_i8 == c_i8;
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : si8
    val eq_i16: bool = c_i16 == c_i16;
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : si16
    val eq_i32: bool = c_i32 == c_i32;
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : si32
    val eq_i64: bool = c_i64 == c_i64;
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : si64
    val eq_u8: bool = c_u8 == c_u8;
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val eq_u16: bool = c_u16 == c_u16;
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val eq_u32: bool = c_u32 == c_u32;
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val eq_u64: bool = c_u64 == c_u64;
#CHECK: {{%[^ ]+}} = arc.cmpi eq, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val eq_f32: bool = c_f32 == c_f32;
#CHECK: {{%[^ ]+}} = arith.cmpf oeq, {{%[^ ]+}}, {{%[^ ]+}} : f32
    val eq_f64: bool = c_f64 == c_f64;
#CHECK: {{%[^ ]+}} = arith.cmpf oeq, {{%[^ ]+}}, {{%[^ ]+}} : f64

    val ne_bool: bool = c_bool != c_bool;
#CHECK: {{%[^ ]+}} = arith.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : i1
    val ne_i8: bool = c_i8 != c_i8;
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : si8
    val ne_i16: bool = c_i16 != c_i16;
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : si16
    val ne_i32: bool = c_i32 != c_i32;
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : si32
    val ne_i64: bool = c_i64 != c_i64;
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : si64
    val ne_u8: bool = c_u8 != c_u8;
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val ne_u16: bool = c_u16 != c_u16;
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val ne_u32: bool = c_u32 != c_u32;
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val ne_u64: bool = c_u64 != c_u64;
#CHECK: {{%[^ ]+}} = arc.cmpi ne, {{%[^ ]+}}, {{%[^ ]+}} : ui64
    val ne_f32: bool = c_f32 != c_f32;
#CHECK: {{%[^ ]+}} = arith.cmpf one, {{%[^ ]+}}, {{%[^ ]+}} : f32
    val ne_f64: bool = c_f64 != c_f64;
#CHECK: {{%[^ ]+}} = arith.cmpf one, {{%[^ ]+}}, {{%[^ ]+}} : f64

    val land: bool = c_bool and c_bool;
#CHECK: {{%[^ ]+}} = arith.andi {{%[^ ]+}}, {{%[^ ]+}} : i1
    val lor: bool = c_bool or c_bool;
#CHECK: {{%[^ ]+}} = arith.ori {{%[^ ]+}}, {{%[^ ]+}} : i1
    val lxor: bool = c_bool xor c_bool;
#CHECK: {{%[^ ]+}} = arith.xori {{%[^ ]+}}, {{%[^ ]+}} : i1

    val band_i8: i8 = c_i8 band c_i8;
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : si8
    val band_i16: i16 = c_i16 band c_i16;
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : si16
    val band_i32: i32 = c_i32 band c_i32;
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : si32
    val band_i64: i64 = c_i64 band c_i64;
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : si64
    val band_u8: u8 = c_u8 band c_u8;
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val band_u16: u16 = c_u16 band c_u16;
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val band_u32: u32 = c_u32 band c_u32;
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val band_u64: u64 = c_u64 band c_u64;
#CHECK: {{%[^ ]+}} = arc.and {{%[^ ]+}}, {{%[^ ]+}} : ui64

    val bor_i8: i8 = c_i8 bor c_i8;
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : si8
    val bor_i16: i16 = c_i16 bor c_i16;
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : si16
    val bor_i32: i32 = c_i32 bor c_i32;
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : si32
    val bor_i64: i64 = c_i64 bor c_i64;
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : si64
    val bor_u8: u8 = c_u8 bor c_u8;
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val bor_u16: u16 = c_u16 bor c_u16;
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val bor_u32: u32 = c_u32 bor c_u32;
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val bor_u64: u64 = c_u64 bor c_u64;
#CHECK: {{%[^ ]+}} = arc.or {{%[^ ]+}}, {{%[^ ]+}} : ui64

    val bxor_i8: i8 = c_i8 bxor c_i8;
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : si8
    val bxor_i16: i16 = c_i16 bxor c_i16;
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : si16
    val bxor_i32: i32 = c_i32 bxor c_i32;
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : si32
    val bxor_i64: i64 = c_i64 bxor c_i64;
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : si64
    val bxor_u8: u8 = c_u8 bxor c_u8;
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : ui8
    val bxor_u16: u16 = c_u16 bxor c_u16;
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : ui16
    val bxor_u32: u32 = c_u32 bxor c_u32;
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : ui32
    val bxor_u64: u64 = c_u64 bxor c_u64;
#CHECK: {{%[^ ]+}} = arc.xor {{%[^ ]+}}, {{%[^ ]+}} : ui64

##val min_i8: i8 = min(c_i8, c1_i8);
###XCHECK-DAG: [[A0:%[^ ]+]] = arc.cmpi lt, [[B0:%[^ ]+]], [[C0:%[^ ]+]] : si8
###XCHECK: {{%[^ ]+}} = arc.select [[A0]], [[B0]], [[C0]] : si8
##
##val min_i16: i16 = min(c_i16, c1_i16);
###XCHECK-DAG: [[A1:%[^ ]+]] = arc.cmpi lt, [[B1:%[^ ]+]], [[C1:%[^ ]+]] : si16
###XCHECK: {{%[^ ]+}} = arc.select [[A1]], [[B1]], [[C1]] : si16
##
##val min_i32: i32 = min(c_i32, c1_i32);
###XCHECK-DAG: [[A2:%[^ ]+]] = arc.cmpi lt, [[B2:%[^ ]+]], [[C2:%[^ ]+]] : si32
###XCHECK: {{%[^ ]+}} = arc.select [[A2]], [[B2]], [[C2]] : si32
##
##val min_i64: i64 = min(c_i64, c1_i64);
###XCHECK-DAG: [[A3:%[^ ]+]] = arc.cmpi lt, [[B3:%[^ ]+]], [[C3:%[^ ]+]] : si64
###XCHECK: {{%[^ ]+}} = arc.select [[A3]], [[B3]], [[C3]] : si64
##
##val min_u8: u8 = min(c_u8, c1_u8);
###XCHECK-DAG: [[A4:%[^ ]+]] = arc.cmpi lt, [[B4:%[^ ]+]], [[C4:%[^ ]+]] : ui8
###XCHECK: {{%[^ ]+}} = arc.select [[A4]], [[B4]], [[C4]] : ui8
##
##val min_u16: u16 = min(c_u16, c1_u16);
###XCHECK-DAG: [[A5:%[^ ]+]] = arc.cmpi lt, [[B5:%[^ ]+]], [[C5:%[^ ]+]] : ui16
###XCHECK: {{%[^ ]+}} = arc.select [[A5]], [[B5]], [[C5]] : ui16
##
##val min_u32: u32 = min(c_u32, c1_u32);
###XCHECK-DAG: [[A6:%[^ ]+]] = arc.cmpi lt, [[B6:%[^ ]+]], [[C6:%[^ ]+]] : ui32
###XCHECK: {{%[^ ]+}} = arc.select [[A6]], [[B6]], [[C6]] : ui32
##
##val min_u64: u64 = min(c_u64, c1_u64);
###XCHECK-DAG: [[A7:%[^ ]+]] = arc.cmpi lt, [[B7:%[^ ]+]], [[C7:%[^ ]+]] : ui64
###XCHECK: {{%[^ ]+}} = arc.select [[A7]], [[B7]], [[C7]] : ui64
##
##val min_f32: f32 = min(c_f32, c1_f32);
###XCHECK-DAG: [[A8:%[^ ]+]] = arith.cmpf olt, [[B8:%[^ ]+]], [[C8:%[^ ]+]] : f32
###XCHECK: {{%[^ ]+}} = select [[A8]], [[B8]], [[C8]] : f32
##
##val min_f64: f64 = min(c_f64, c1_f64);
###XCHECK-DAG: [[A9:%[^ ]+]] = arith.cmpf olt, [[B9:%[^ ]+]], [[C9:%[^ ]+]] : f64
###XCHECK: {{%[^ ]+}} = select [[A9]], [[B9]], [[C9]] : f64
##
##val max_i8: i8 = max(c_i8, c1_i8);
###XCHECK-DAG: [[A0:%[^ ]+]] = arc.cmpi lt, [[B0:%[^ ]+]], [[C0:%[^ ]+]] : si8
###XCHECK: {{%[^ ]+}} = arc.select [[A0]], [[C0]], [[B0]] : si8
##
##val max_i16: i16 = max(c_i16, c1_i16);
###XCHECK-DAG: [[A1:%[^ ]+]] = arc.cmpi lt, [[B1:%[^ ]+]], [[C1:%[^ ]+]] : si16
###XCHECK: {{%[^ ]+}} = arc.select [[A1]], [[C1]], [[B1]] : si16
##
##val max_i32: i32 = max(c_i32, c1_i32);
###XCHECK-DAG: [[A2:%[^ ]+]] = arc.cmpi lt, [[B2:%[^ ]+]], [[C2:%[^ ]+]] : si32
###XCHECK: {{%[^ ]+}} = arc.select [[A2]], [[C2]], [[B2]] : si32
##
##val max_i64: i64 = max(c_i64, c1_i64);
###XCHECK-DAG: [[A3:%[^ ]+]] = arc.cmpi lt, [[B3:%[^ ]+]], [[C3:%[^ ]+]] : si64
###XCHECK: {{%[^ ]+}} = arc.select [[A3]], [[C3]], [[B3]] : si64
##
##val max_u8: u8 = max(c_u8, c1_u8);
###XCHECK-DAG: [[A4:%[^ ]+]] = arc.cmpi lt, [[B4:%[^ ]+]], [[C4:%[^ ]+]] : ui8
###XCHECK: {{%[^ ]+}} = arc.select [[A4]], [[C4]], [[B4]] : ui8
##
##val max_u16: u16 = max(c_u16, c1_u16);
###XCHECK-DAG: [[A5:%[^ ]+]] = arc.cmpi lt, [[B5:%[^ ]+]], [[C5:%[^ ]+]] : ui16
###XCHECK: {{%[^ ]+}} = arc.select [[A5]], [[C5]], [[B5]] : ui16
##
##val max_u32: u32 = max(c_u32, c1_u32);
###XCHECK-DAG: [[A6:%[^ ]+]] = arc.cmpi lt, [[B6:%[^ ]+]], [[C6:%[^ ]+]] : ui32
###XCHECK: {{%[^ ]+}} = arc.select [[A6]], [[C6]], [[B6]] : ui32
##
##val max_u64: u64 = max(c_u64, c1_u64);
###XCHECK-DAG: [[A7:%[^ ]+]] = arc.cmpi lt, [[B7:%[^ ]+]], [[C7:%[^ ]+]] : ui64
###XCHECK: {{%[^ ]+}} = arc.select [[A7]], [[C7]], [[B7]] : ui64
##
##val max_f32: f32 = max(c_f32, c1_f32);
###XCHECK-DAG: [[A8:%[^ ]+]] = arith.cmpf olt, [[B8:%[^ ]+]], [[C8:%[^ ]+]] : f32
###XCHECK: {{%[^ ]+}} = select [[A8]], [[C8]], [[B8]] : f32
##
##val max_f64: f64 = max(c_f64, c1_f64);
###XCHECK-DAG: [[A9:%[^ ]+]] = arith.cmpf olt, [[B9:%[^ ]+]], [[C9:%[^ ]+]] : f64
###XCHECK: {{%[^ ]+}} = select [[A9]], [[C9]], [[B9]] : f64

    val pow_f32: f32 = c_f32 ** c1_f32;
#CHECK: {{%[^ ]+}} = math.powf {{%[^ ]+}}, {{%[^ ]+}} : f32

    val pow_f64: f64 = c_f64 ** c1_f64;
#CHECK: {{%[^ ]+}} = math.powf {{%[^ ]+}}, {{%[^ ]+}} : f64

}
