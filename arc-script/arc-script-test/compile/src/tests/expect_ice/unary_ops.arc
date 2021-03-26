# RUN: arc-to-mlir -i %s | FileCheck %s
# RUN: arc-to-mlir -i %s | arc-mlir | FileCheck %s

fun test(): i32 {
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

    let not_bool: bool = not c_bool in
#CHECK: {{%[^ ]+}} = not {{%[^ ]+}} : bool

    let exp_f32: f32 = exp(c_f32) in
#CHECK: {{%[^ ]+}} = exp {{%[^ ]+}} : f32

    let exp_f64: f64 = exp(c_f64) in
#CHECK: {{%[^ ]+}} = exp {{%[^ ]+}} : f64

    let log_f32: f32 = log(c_f32) in
#CHECK: {{%[^ ]+}} = log {{%[^ ]+}} : f32

    let log_f64: f64 = log(c_f64) in
#CHECK: {{%[^ ]+}} = log {{%[^ ]+}} : f64

    let cos_f32: f32 = cos(c_f32) in
#CHECK: {{%[^ ]+}} = cos {{%[^ ]+}} : f32

    let cos_f64: f64 = cos(c_f64) in
#CHECK: {{%[^ ]+}} = cos {{%[^ ]+}} : f64

    let sin_f32: f32 = sin(c_f32) in
#CHECK: {{%[^ ]+}} = sin {{%[^ ]+}} : f32

    let sin_f64: f64 = sin(c_f64) in
#CHECK: {{%[^ ]+}} = sin {{%[^ ]+}} : f64

    let tan_f32: f32 = tan(c_f32) in
#CHECK: {{%[^ ]+}} = "arc.tan"({{%[^ ]+}}): (f32) -> f32

    let tan_f64: f64 = tan(c_f64) in
#CHECK: {{%[^ ]+}} = "arc.tan"({{%[^ ]+}}): (f64) -> f64

    let acos_f32: f32 = acos(c_f32) in
#CHECK: {{%[^ ]+}} = "arc.acos"({{%[^ ]+}}) : (f32) -> f32

    let acos_f64: f64 = acos(c_f64) in
#CHECK: {{%[^ ]+}} = "arc.acos"({{%[^ ]+}}) : (f64) -> f64

    let asin_f32: f32 = asin(c_f32) in
#CHECK: {{%[^ ]+}} = "arc.asin"({{%[^ ]+}}) : (f32) -> f32

    let asin_f64: f64 = asin(c_f64) in
#CHECK: {{%[^ ]+}} = "arc.asin"({{%[^ ]+}}) : (f64) -> f64

    let atan_f32: f32 = atan(c_f32) in
#CHECK: {{%[^ ]+}} =  atan {{%[^ ]+}} : f32

    let atan_f64: f64 = atan(c_f64) in
#CHECK: {{%[^ ]+}} = atan {{%[^ ]+}} : f64

    let cosh_f32: f32 = cosh(c_f32) in
#CHECK: {{%[^ ]+}} = "arc.cosh"({{%[^ ]+}}) : (f32) -> f32

    let cosh_f64: f64 = cosh(c_f64) in
#CHECK: {{%[^ ]+}} = "arc.cosh"({{%[^ ]+}}) : (f64) -> f64

    let sinh_f32: f32 = sinh(c_f32) in
#CHECK: {{%[^ ]+}} = "arc.sinh"({{%[^ ]+}}) : (f32) -> f32

    let sinh_f64: f64 = sinh(c_f64) in
#CHECK: {{%[^ ]+}} = "arc.sinh"({{%[^ ]+}}) : (f64) -> f64

    let tanh_f32: f32 = tanh(c_f32) in
#CHECK: {{%[^ ]+}} = tanh {{%[^ ]+}} : f32

    let tanh_f64: f64 = tanh(c_f64) in
#CHECK: {{%[^ ]+}} = tanh {{%[^ ]+}} : f64

    let erf_f32: f32 = erf(c_f32) in
#CHECK: {{%[^ ]+}} = "arc.erf"({{%[^ ]+}}) : (f32) -> f32

    let erf_f64: f64 = erf(c_f64) in
#CHECK: {{%[^ ]+}} = "arc.erf"({{%[^ ]+}}) : (f64) -> f64

    let sqrt_f32: f32 = sqrt(c_f32) in
#CHECK: {{%[^ ]+}} = sqrt {{%[^ ]+}} : f32

    let sqrt_f64: f64 = sqrt(c_f64) in
#CHECK: {{%[^ ]+}} = sqrt {{%[^ ]+}} : f64

    4711
}
