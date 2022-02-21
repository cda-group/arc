# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

def test() {
# ANCHOR: example
val v0: i32             = 3;                   # Integer
val v1: f32             = 0.1;                 # Float
val v2: f32             = 10%;                 # Float (Percentage)
val v3: i32             = 100ms;               # Duration
val v4: String          = 2020-12-16T16:00:00; # DateTime
val v5: char            = 'c';                 # Character
val v6: String          = "hello";             # String
val v7: String          = "$v6 world";         # String (Interpolated)
val v8: #{x:i32, y:i32} = #{x:5, y:8};         # Record
val v9: Option[i32]     = Option::Some(3);     # Enum variant
val v10: [i32]          = [1,2,3];             # Vector
val v11: fun(i32): i32  = fun(x:i32) = x;      # Lambda function
# ANCHOR_END: example
}
