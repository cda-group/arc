let xi8   = 1i8;
let xi16  = 1i16;
let xi32  = 1;
let xi64  = 1i64;
let xf32  = 1.1f32;
let xf64  = 1.1;
let xbool = true;
1

# args: --check file
# expected stdout:
# let xi8: i8 = 1i8;
# let xi16: i16 = 1i16;
# let xi32: i32 = 1;
# let xi64: i64 = 1i64;
# let xf32: f32 = 1.1f32;
# let xf64: f64 = 1.1;
# let xbool: bool = true;
# let x7: i32 = 1;
# x7
