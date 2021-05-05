# RUN: arc-to-mlir -i %s | FileCheck %s
# RUN: arc-to-mlir -i %s | arc-mlir | FileCheck %s

fun test() {
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

    val not_bool: bool = not c_bool;
#CHECK: {{%[^ ]+}} = not {{%[^ ]+}} : bool

    val exp_f32: f32 = exp(c_f32);
#CHECK: {{%[^ ]+}} = exp {{%[^ ]+}} : f32

    val exp_f64: f64 = exp(c_f64);
#CHECK: {{%[^ ]+}} = exp {{%[^ ]+}} : f64

    val log_f32: f32 = log(c_f32);
#CHECK: {{%[^ ]+}} = log {{%[^ ]+}} : f32

    val log_f64: f64 = log(c_f64);
#CHECK: {{%[^ ]+}} = log {{%[^ ]+}} : f64

    val cos_f32: f32 = cos(c_f32);
#CHECK: {{%[^ ]+}} = cos {{%[^ ]+}} : f32

    val cos_f64: f64 = cos(c_f64);
#CHECK: {{%[^ ]+}} = cos {{%[^ ]+}} : f64

    val sin_f32: f32 = sin(c_f32);
#CHECK: {{%[^ ]+}} = sin {{%[^ ]+}} : f32

    val sin_f64: f64 = sin(c_f64);
#CHECK: {{%[^ ]+}} = sin {{%[^ ]+}} : f64

    val tan_f32: f32 = tan(c_f32);
#CHECK: {{%[^ ]+}} = "arc.tan"({{%[^ ]+}}): (f32) -> f32

    val tan_f64: f64 = tan(c_f64);
#CHECK: {{%[^ ]+}} = "arc.tan"({{%[^ ]+}}): (f64) -> f64

    val acos_f32: f32 = acos(c_f32);
#CHECK: {{%[^ ]+}} = "arc.acos"({{%[^ ]+}}) : (f32) -> f32

    val acos_f64: f64 = acos(c_f64);
#CHECK: {{%[^ ]+}} = "arc.acos"({{%[^ ]+}}) : (f64) -> f64

    val asin_f32: f32 = asin(c_f32);
#CHECK: {{%[^ ]+}} = "arc.asin"({{%[^ ]+}}) : (f32) -> f32

    val asin_f64: f64 = asin(c_f64);
#CHECK: {{%[^ ]+}} = "arc.asin"({{%[^ ]+}}) : (f64) -> f64

    val atan_f32: f32 = atan(c_f32);
#CHECK: {{%[^ ]+}} =  atan {{%[^ ]+}} : f32

    val atan_f64: f64 = atan(c_f64);
#CHECK: {{%[^ ]+}} = atan {{%[^ ]+}} : f64

    val cosh_f32: f32 = cosh(c_f32);
#CHECK: {{%[^ ]+}} = "arc.cosh"({{%[^ ]+}}) : (f32) -> f32

    val cosh_f64: f64 = cosh(c_f64);
#CHECK: {{%[^ ]+}} = "arc.cosh"({{%[^ ]+}}) : (f64) -> f64

    val sinh_f32: f32 = sinh(c_f32);
#CHECK: {{%[^ ]+}} = "arc.sinh"({{%[^ ]+}}) : (f32) -> f32

    val sinh_f64: f64 = sinh(c_f64);
#CHECK: {{%[^ ]+}} = "arc.sinh"({{%[^ ]+}}) : (f64) -> f64

    val tanh_f32: f32 = tanh(c_f32);
#CHECK: {{%[^ ]+}} = tanh {{%[^ ]+}} : f32

    val tanh_f64: f64 = tanh(c_f64);
#CHECK: {{%[^ ]+}} = tanh {{%[^ ]+}} : f64

    val erf_f32: f32 = erf(c_f32);
#CHECK: {{%[^ ]+}} = "arc.erf"({{%[^ ]+}}) : (f32) -> f32

    val erf_f64: f64 = erf(c_f64);
#CHECK: {{%[^ ]+}} = "arc.erf"({{%[^ ]+}}) : (f64) -> f64

    val sqrt_f32: f32 = sqrt(c_f32);
#CHECK: {{%[^ ]+}} = sqrt {{%[^ ]+}} : f32

    val sqrt_f64: f64 = sqrt(c_f64);
#CHECK: {{%[^ ]+}} = sqrt {{%[^ ]+}} : f64
}
