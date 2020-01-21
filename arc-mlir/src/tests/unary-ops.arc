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

let exp_f32 : f32 = exp c_f32;
#CHECK: {{%[^ ]+}} = exp {{%[^ ]+}} : f32

let exp_f64 : f64 = exp c_f64;
#CHECK: {{%[^ ]+}} = exp {{%[^ ]+}} : f64

let log_f32 : f32 = log c_f32;
#CHECK: {{%[^ ]+}} = log {{%[^ ]+}} : f32

let log_f64 : f64 = log c_f64;
#CHECK: {{%[^ ]+}} = log {{%[^ ]+}} : f64

let cos_f32 : f32 = cos c_f32;
#CHECK: {{%[^ ]+}} = cos {{%[^ ]+}} : f32

let cos_f64 : f64 = cos c_f64;
#CHECK: {{%[^ ]+}} = cos {{%[^ ]+}} : f64

let sin_f32 : f32 = sin c_f32;
#CHECK: {{%[^ ]+}} = "arc.sin"({{%[^ ]+}}) : (f32) -> f32

let sin_f64 : f64 = sin c_f64;
#CHECK: {{%[^ ]+}} = "arc.sin"({{%[^ ]+}}) : (f64) -> f64

let tan_f32 : f32 = tan c_f32;
#CHECK: {{%[^ ]+}} = "arc.tan"({{%[^ ]+}}) : (f32) -> f32

let tan_f64 : f64 = tan c_f64;
#CHECK: {{%[^ ]+}} = "arc.tan"({{%[^ ]+}}) : (f64) -> f64

4711
