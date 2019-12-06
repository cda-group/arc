# RUN: arc-to-mlir -i %s | FileCheck %s
# RUN: arc-to-mlir -i %s | arc-mlir | FileCheck %s

# Here we check positive literals, there is no point in checking the
# negative literals as they will be represented by a
# negation. Likewise, as weld does not have a syntax for unsigned
# literals we don't check that either.

let pos_i8 : i8 = 127c;
#CHECK: {{%[^ ]+}} = constant 127 : i8

let neg_i8 : i8 = -128c;
#CHECK: {{%[^ ]+}} = constant -128 : i8

let pos_i16 : i16 = 32767si;
#CHECK: {{%[^ ]+}} = constant 32767 : i16

let neg_i16 : i16 = -32768si;
#CHECK: {{%[^ ]+}} = constant -32768 : i16

let pos_i32 : i32 = 2147483647;
#CHECK: {{%[^ ]+}} = constant 2147483647 : i32

let neg_i32 : i32 = -2147483648;
#CHECK: {{%[^ ]+}} = constant -2147483648 : i32

let pos_i64 : i64 = 9223372036854775807L;
#CHECK: {{%[^ ]+}} = constant 9223372036854775807 : i64

let neg_i64 : i64 = -9223372036854775808L;
#CHECK: {{%[^ ]+}} = constant -9223372036854775808 : i64

# As MLIR does not support hex floating point values we only check the
# f32 to 6 significant digits and f64 to 15.

let pos_f32 : f32 = 3.4028234664e38f;
#CHECK: {{%[^ ]+}} = constant 3.40282{{[0-9]+[Ee]\+?}}38 : f32

let neg_f32 : f32 = -3.4028234664e38f;
#CHECK: {{%[^ ]+}} = constant -3.40282{{[0-9]+[Ee]\+?}}38 : f32

let pos_f64 : f64 = 1.7976931348623157e308;
#CHECK: {{%[^ ]+}} = constant 1.79769313486231{{[0-9]+[Ee]\+?}}308 : f64

let neg_f64 : f64 = -1.7976931348623157e308;
#CHECK: {{%[^ ]+}} = constant -1.79769313486231{{[0-9]+[Ee]\+?}}308 : f64

let true_bool : bool = true;
#CHECK: {{%[^ ]+}} = constant 1 : i1

let false_bool : bool = false;
#CHECK: {{%[^ ]+}} = constant 0 : i1

let bool_vector : vec[bool] = [true, false, true, false];
#CHECK-DAG: [[E0:%[^ ]+]] = constant 1 : i1
#CHECK-DAG: [[E1:%[^ ]+]] = constant 0 : i1
#CHECK-DAG: [[E2:%[^ ]+]] = constant 1 : i1
#CHECK-DAG: [[E3:%[^ ]+]] = constant 0 : i1
#CHECK: {{%[^ ]+}} = "arc.make_vector"([[E0]], [[E1]], [[E2]], [[E3]]) : (i1, i1, i1, i1) -> tensor<4xi1>

let f64_vector : vec[f64] = [0.694, 1.0, 1.4142, 3.14];
#CHECK-DAG: [[E4:%[^ ]+]] = constant {{[^:]+}} : f64
#CHECK-DAG: [[E5:%[^ ]+]] = constant {{[^:]+}} : f64
#CHECK-DAG: [[E6:%[^ ]+]] = constant {{[^:]+}} : f64
#CHECK-DAG: [[E7:%[^ ]+]] = constant {{[^:]+}} : f64
#CHECK: {{%[^ ]+}} = "arc.make_vector"([[E4]], [[E5]], [[E6]], [[E7]]) : (f64, f64, f64, f64) -> tensor<4xf64>

4711
