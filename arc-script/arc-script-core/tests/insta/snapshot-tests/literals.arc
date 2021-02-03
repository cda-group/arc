# RUN: arc-script run --output=MLIR %s | FileCheck %s
# RUN: arc-script run --output=MLIR %s | arc-mlir | FileCheck %s
# RUN: arc-script run --output=MLIR %s | arc-mlir | -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

fun test() -> i32 {

  let pos_i8: i8 = 127i8 in
#CHECK: {{%[^ ]+}} = arc.constant 127 : si8
  let neg_i8: i8 = -128i8 in
#CHECK: {{%[^ ]+}} = arc.constant -128 : si8
  let pos_i16: i16 = 32767i16 in
#CHECK: {{%[^ ]+}} = arc.constant 32767 : si16
  let neg_i16: i16 = -32768i16 in
#CHECK: {{%[^ ]+}} = arc.constant -32768 : si16
  let pos_i32: i32 = 2147483647 in
#CHECK: {{%[^ ]+}} = arc.constant 2147483647 : si32
  let neg_i32: i32 = -2147483648 in
#CHECK: {{%[^ ]+}} = arc.constant -2147483648 : si32
  let pos_i64: i64 = 9223372036854775807i64 in
#CHECK: {{%[^ ]+}} = arc.constant 9223372036854775807 : si64
  let neg_i64: i64 = -9223372036854775808i64 in
#CHECK: {{%[^ ]+}} = arc.constant -9223372036854775808 : si64
  let pos_u8: u8 = 255u8 in
#CHECK: {{%[^ ]+}} = arc.constant 255 : ui8
  let pos_u16: u16 = 65535u16 in
#CHECK: {{%[^ ]+}} = arc.constant 65535 : ui16

  let pos_u32: u32 = 4294967295u32 in
#CHECK: {{%[^ ]+}} = arc.constant 4294967295 : ui32

  let pos_u64: u64 = 18446744073709551615u64 in
#CHECK: {{%[^ ]+}} = arc.constant 18446744073709551615 : ui64

# As MLIR does not support hex floating point values we only check the
# f32 to 6 significant digits and f64 to 15.

  let pos_bf16: bf16 = 3.38953139e38bf16 in
#CHECK: {{%[^ ]+}} = constant 3.38953139{{[0-9]+[Ee]\+?}}38 : bf16

  let neg_bf16: bf16 = -1.175494351e38bf16 in
#CHECK: {{%[^ ]+}} = constant -1.175494351{{[0-9]+[Ee]\+?}}38 : bf16

  let pos_f16: f16 = 6.5504e4f16 in
#CHECK: {{%[^ ]+}} = constant 6.55{{[0-9]+[Ee]\+?}}4 : f16

  let neg_f16: f16 = -6.550e4f16 in
#CHECK: {{%[^ ]+}} = constant -6.550{{[0-9]+[Ee]\+?}}4 : f16

  let pos_f32: f32 = 3.4028234664e38f32 in
#CHECK: {{%[^ ]+}} = constant 3.40282{{[0-9]+[Ee]\+?}}38 : f32

  let neg_f32: f32 = -3.4028234664e38f32 in
#CHECK: {{%[^ ]+}} = constant -3.40282{{[0-9]+[Ee]\+?}}38 : f32

  let pos_f64: f64 = 1.7976931348623157e308 in
#CHECK: {{%[^ ]+}} = constant 1.79769313486231{{[0-9]+[Ee]\+?}}308 : f64

  let neg_f64: f64 = -1.7976931348623157e308 in
#CHECK: {{%[^ ]+}} = constant -1.79769313486231{{[0-9]+[Ee]\+?}}308 : f64

  let true_bool: bool = true in
#CHECK: {{%[^ ]+}} = constant true

  let false_bool: bool = false in
#CHECK: {{%[^ ]+}} = constant false

#  let bool_vector: vec[bool] = [true, false, true, false] in
##CHECK-DAG: [[E0:%[^ ]+]] = constant true
##CHECK-DAG: [[E1:%[^ ]+]] = constant false
##CHECK-DAG: [[E2:%[^ ]+]] = constant true
##CHECK-DAG: [[E3:%[^ ]+]] = constant false
##CHECK: {{%[^ ]+}} = "arc.make_vector"([[E0]], [[E1]], [[E2]], [[E3]]) : (i1, i1, i1, i1) -> tensor<4xi1>
#
#  let f64_vector: vec[f64] = [0.694, 1.0, 1.4142, 3.14] in
##CHECK-DAG: [[E4:%[^ ]+]] = constant {{[^:]+}} : f64
##CHECK-DAG: [[E5:%[^ ]+]] = constant {{[^:]+}} : f64
##CHECK-DAG: [[E6:%[^ ]+]] = constant {{[^:]+}} : f64
##CHECK-DAG: [[E7:%[^ ]+]] = constant {{[^:]+}} : f64
##CHECK: {{%[^ ]+}} = "arc.make_vector"([[E4]], [[E5]], [[E6]], [[E7]]) : (f64, f64, f64, f64) -> tensor<4xf64>
  4711

}
