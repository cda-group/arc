# RUN: arc-script --no-prelude run --output=MLIR %s | FileCheck %s
# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir | FileCheck %s
# RUNX: arc-script --no-prelude run --output=MLIR %s | arc-mlir | arc-mlir -rustcratename expectpassliterals -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/expectpassliterals/Cargo.toml

fun test() {

  val pos_i8: i8 = 127i8;
#CHECK: {{%[^ ]+}} = arc.constant 127 : si8
  val neg_i8: i8 = -128i8;
#CHECK: {{%[^ ]+}} = arc.constant -128 : si8
  val pos_i16: i16 = 32767i16;
#CHECK: {{%[^ ]+}} = arc.constant 32767 : si16
  val neg_i16: i16 = -32768i16;
#CHECK: {{%[^ ]+}} = arc.constant -32768 : si16
  val pos_i32: i32 = 2147483647;
#CHECK: {{%[^ ]+}} = arc.constant 2147483647 : si32
  val neg_i32: i32 = -2147483648;
#CHECK: {{%[^ ]+}} = arc.constant -2147483648 : si32
  val pos_i64: i64 = 9223372036854775807i64;
#CHECK: {{%[^ ]+}} = arc.constant 9223372036854775807 : si64
  val neg_i64: i64 = -9223372036854775808i64;
#CHECK: {{%[^ ]+}} = arc.constant -9223372036854775808 : si64
  val pos_u8: u8 = 255u8;
#CHECK: {{%[^ ]+}} = arc.constant 255 : ui8
  val pos_u16: u16 = 65535u16;
#CHECK: {{%[^ ]+}} = arc.constant 65535 : ui16

  val pos_u32: u32 = 4294967295u32;
#CHECK: {{%[^ ]+}} = arc.constant 4294967295 : ui32

  val pos_u64: u64 = 18446744073709551615u64;
#CHECK: {{%[^ ]+}} = arc.constant 18446744073709551615 : ui64

# As MLIR does not support hex floating point values we only check the
# f32 to 6 significant digits and f64 to 15.

  val pos_f32: f32 = 3.4028234664e38f32;
#CHECK: {{%[^ ]+}} = arith.constant 3.40282{{[0-9]+[Ee]\+?}}38 : f32

  val neg_f32: f32 = -3.4028234664e38f32;
#CHECK: {{%[^ ]+}} = arith.constant -3.40282{{[0-9]+[Ee]\+?}}38 : f32

  val pos_f64: f64 = 1.7976931348623157e308;
#CHECK: {{%[^ ]+}} = arith.constant 1.79769313486231{{[0-9]+[Ee]\+?}}308 : f64

  val neg_f64: f64 = -1.7976931348623157e308;
#CHECK: {{%[^ ]+}} = arith.constant -1.79769313486231{{[0-9]+[Ee]\+?}}308 : f64

  val true_bool: bool = true;
#CHECK: {{%[^ ]+}} = arith.constant true

  val false_bool: bool = false;
#CHECK: {{%[^ ]+}} = arith.constant false

#  val bool_vector: vec[bool] = [true, false, true, false];
##XCHECK-DAG: [[E0:%[^ ]+]] = arith.constant true
##XCHECK-DAG: [[E1:%[^ ]+]] = arith.constant false
##XCHECK-DAG: [[E2:%[^ ]+]] = arith.constant true
##XCHECK-DAG: [[E3:%[^ ]+]] = arith.constant false
##XCHECK: {{%[^ ]+}} = "arc.make_vector"([[E0]], [[E1]], [[E2]], [[E3]]) : (i1, i1, i1, i1) -> tensor<4xi1>
#
#  val f64_vector: vec[f64] = [0.694, 1.0, 1.4142, 3.14];
##XCHECK-DAG: [[E4:%[^ ]+]] = arith.constant {{[^:]+}} : f64
##XCHECK-DAG: [[E5:%[^ ]+]] = arith.constant {{[^:]+}} : f64
##XCHECK-DAG: [[E6:%[^ ]+]] = arith.constant {{[^:]+}} : f64
##XCHECK-DAG: [[E7:%[^ ]+]] = arith.constant {{[^:]+}} : f64
##XCHECK: {{%[^ ]+}} = "arc.make_vector"([[E4]], [[E5]], [[E6]], [[E7]]) : (f64, f64, f64, f64) -> tensor<4xf64>

}
