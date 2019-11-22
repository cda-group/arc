# RUN: arc-to-mlir -i %s | FileCheck %s

# Here we check positive literals, there is no point in checking the
# negative literals as they will be represented by a
# negation. Likewise, as weld does not have a syntax for unsigned
# literals we don't check that either.

let pos_i8 : i8 = 127c;
#CHECK: {{%[^ ]+}} = constant 127 : i8

let pos_i16 : i16 = 32767si;
#CHECK: {{%[^ ]+}} = constant 32767 : i16

let pos_i32 : i32 = 2147483647;
#CHECK: {{%[^ ]+}} = constant 2147483647 : i32

let pos_i64 : i64 = 9223372036854775807L;
#CHECK: {{%[^ ]+}} = constant 9223372036854775807 : i64

4711
