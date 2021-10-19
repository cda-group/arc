// RUN: arc-mlir --canonicalize %s | FileCheck %s
module @toplevel {
  func @main(%arg0 : ui64) {
    %true = arith.constant 1 : i1
    %false = arith.constant 0 : i1
    // CHECK-DAG: [[TRUE:%[^ ]+]] = arith.constant true
    // CHECK-DAG: [[FALSE:%[^ ]+]] = arith.constant false
    %smaller_si8 = arc.constant -64 : si8
    %larger_si8 = arc.constant 64 : si8
    %smaller_ui8 = arc.constant 28 : ui8
    %larger_ui8 = arc.constant 128 : ui8
    %smaller_si16 = arc.constant -16384 : si16
    %larger_si16 = arc.constant 16384 : si16
    %smaller_ui16 = arc.constant 32668 : ui16
    %larger_ui16 = arc.constant 32768 : ui16
    %smaller_si32 = arc.constant -1073741824 : si32
    %larger_si32 = arc.constant 1073741824 : si32
    %smaller_ui32 = arc.constant 2147483548 : ui32
    %larger_ui32 = arc.constant 2147483648 : ui32
    %smaller_si64 = arc.constant -4611686018427387904 : si64
    %larger_si64 = arc.constant 4611686018427387904 : si64
    %smaller_ui64 = arc.constant 9223372036854775708 : ui64
    %larger_ui64 = arc.constant 9223372036854775808 : ui64
// CHECK-DAG: [[LARGERU64:%[^ ]+]] = arc.constant 9223372036854775808 : ui64
    %test_eq_ui8_smaller_ui8_smaller_ui8 = arc.cmpi "eq", %smaller_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_eq_ui8_smaller_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_ui8_smaller_ui8_larger_ui8 = arc.cmpi "eq", %smaller_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_eq_ui8_smaller_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_ui8_larger_ui8_smaller_ui8 = arc.cmpi "eq", %larger_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_eq_ui8_larger_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_ui8_larger_ui8_larger_ui8 = arc.cmpi "eq", %larger_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_eq_ui8_larger_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_si8_smaller_si8_smaller_si8 = arc.cmpi "eq", %smaller_si8, %smaller_si8 : si8
    "arc.keep"(%test_eq_si8_smaller_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_si8_smaller_si8_larger_si8 = arc.cmpi "eq", %smaller_si8, %larger_si8 : si8
    "arc.keep"(%test_eq_si8_smaller_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_si8_larger_si8_smaller_si8 = arc.cmpi "eq", %larger_si8, %smaller_si8 : si8
    "arc.keep"(%test_eq_si8_larger_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_si8_larger_si8_larger_si8 = arc.cmpi "eq", %larger_si8, %larger_si8 : si8
    "arc.keep"(%test_eq_si8_larger_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_ui16_smaller_ui16_smaller_ui16 = arc.cmpi "eq", %smaller_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_eq_ui16_smaller_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_ui16_smaller_ui16_larger_ui16 = arc.cmpi "eq", %smaller_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_eq_ui16_smaller_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_ui16_larger_ui16_smaller_ui16 = arc.cmpi "eq", %larger_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_eq_ui16_larger_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_ui16_larger_ui16_larger_ui16 = arc.cmpi "eq", %larger_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_eq_ui16_larger_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_si16_smaller_si16_smaller_si16 = arc.cmpi "eq", %smaller_si16, %smaller_si16 : si16
    "arc.keep"(%test_eq_si16_smaller_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_si16_smaller_si16_larger_si16 = arc.cmpi "eq", %smaller_si16, %larger_si16 : si16
    "arc.keep"(%test_eq_si16_smaller_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_si16_larger_si16_smaller_si16 = arc.cmpi "eq", %larger_si16, %smaller_si16 : si16
    "arc.keep"(%test_eq_si16_larger_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_si16_larger_si16_larger_si16 = arc.cmpi "eq", %larger_si16, %larger_si16 : si16
    "arc.keep"(%test_eq_si16_larger_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_ui32_smaller_ui32_smaller_ui32 = arc.cmpi "eq", %smaller_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_eq_ui32_smaller_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_ui32_smaller_ui32_larger_ui32 = arc.cmpi "eq", %smaller_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_eq_ui32_smaller_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_ui32_larger_ui32_smaller_ui32 = arc.cmpi "eq", %larger_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_eq_ui32_larger_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_ui32_larger_ui32_larger_ui32 = arc.cmpi "eq", %larger_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_eq_ui32_larger_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_si32_smaller_si32_smaller_si32 = arc.cmpi "eq", %smaller_si32, %smaller_si32 : si32
    "arc.keep"(%test_eq_si32_smaller_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_si32_smaller_si32_larger_si32 = arc.cmpi "eq", %smaller_si32, %larger_si32 : si32
    "arc.keep"(%test_eq_si32_smaller_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_si32_larger_si32_smaller_si32 = arc.cmpi "eq", %larger_si32, %smaller_si32 : si32
    "arc.keep"(%test_eq_si32_larger_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_si32_larger_si32_larger_si32 = arc.cmpi "eq", %larger_si32, %larger_si32 : si32
    "arc.keep"(%test_eq_si32_larger_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_ui64_smaller_ui64_smaller_ui64 = arc.cmpi "eq", %smaller_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_eq_ui64_smaller_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_ui64_smaller_ui64_larger_ui64 = arc.cmpi "eq", %smaller_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_eq_ui64_smaller_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_ui64_larger_ui64_smaller_ui64 = arc.cmpi "eq", %larger_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_eq_ui64_larger_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_ui64_larger_ui64_larger_ui64 = arc.cmpi "eq", %larger_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_eq_ui64_larger_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_si64_smaller_si64_smaller_si64 = arc.cmpi "eq", %smaller_si64, %smaller_si64 : si64
    "arc.keep"(%test_eq_si64_smaller_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_eq_si64_smaller_si64_larger_si64 = arc.cmpi "eq", %smaller_si64, %larger_si64 : si64
    "arc.keep"(%test_eq_si64_smaller_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_si64_larger_si64_smaller_si64 = arc.cmpi "eq", %larger_si64, %smaller_si64 : si64
    "arc.keep"(%test_eq_si64_larger_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_eq_si64_larger_si64_larger_si64 = arc.cmpi "eq", %larger_si64, %larger_si64 : si64
    "arc.keep"(%test_eq_si64_larger_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui8_smaller_ui8_smaller_ui8 = arc.cmpi "ne", %smaller_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_ne_ui8_smaller_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_ui8_smaller_ui8_larger_ui8 = arc.cmpi "ne", %smaller_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_ne_ui8_smaller_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui8_larger_ui8_smaller_ui8 = arc.cmpi "ne", %larger_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_ne_ui8_larger_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui8_larger_ui8_larger_ui8 = arc.cmpi "ne", %larger_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_ne_ui8_larger_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_si8_smaller_si8_smaller_si8 = arc.cmpi "ne", %smaller_si8, %smaller_si8 : si8
    "arc.keep"(%test_ne_si8_smaller_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_si8_smaller_si8_larger_si8 = arc.cmpi "ne", %smaller_si8, %larger_si8 : si8
    "arc.keep"(%test_ne_si8_smaller_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_si8_larger_si8_smaller_si8 = arc.cmpi "ne", %larger_si8, %smaller_si8 : si8
    "arc.keep"(%test_ne_si8_larger_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_si8_larger_si8_larger_si8 = arc.cmpi "ne", %larger_si8, %larger_si8 : si8
    "arc.keep"(%test_ne_si8_larger_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_ui16_smaller_ui16_smaller_ui16 = arc.cmpi "ne", %smaller_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_ne_ui16_smaller_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_ui16_smaller_ui16_larger_ui16 = arc.cmpi "ne", %smaller_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_ne_ui16_smaller_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui16_larger_ui16_smaller_ui16 = arc.cmpi "ne", %larger_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_ne_ui16_larger_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui16_larger_ui16_larger_ui16 = arc.cmpi "ne", %larger_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_ne_ui16_larger_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_si16_smaller_si16_smaller_si16 = arc.cmpi "ne", %smaller_si16, %smaller_si16 : si16
    "arc.keep"(%test_ne_si16_smaller_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_si16_smaller_si16_larger_si16 = arc.cmpi "ne", %smaller_si16, %larger_si16 : si16
    "arc.keep"(%test_ne_si16_smaller_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_si16_larger_si16_smaller_si16 = arc.cmpi "ne", %larger_si16, %smaller_si16 : si16
    "arc.keep"(%test_ne_si16_larger_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_si16_larger_si16_larger_si16 = arc.cmpi "ne", %larger_si16, %larger_si16 : si16
    "arc.keep"(%test_ne_si16_larger_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_ui32_smaller_ui32_smaller_ui32 = arc.cmpi "ne", %smaller_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_ne_ui32_smaller_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_ui32_smaller_ui32_larger_ui32 = arc.cmpi "ne", %smaller_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_ne_ui32_smaller_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui32_larger_ui32_smaller_ui32 = arc.cmpi "ne", %larger_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_ne_ui32_larger_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui32_larger_ui32_larger_ui32 = arc.cmpi "ne", %larger_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_ne_ui32_larger_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_si32_smaller_si32_smaller_si32 = arc.cmpi "ne", %smaller_si32, %smaller_si32 : si32
    "arc.keep"(%test_ne_si32_smaller_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_si32_smaller_si32_larger_si32 = arc.cmpi "ne", %smaller_si32, %larger_si32 : si32
    "arc.keep"(%test_ne_si32_smaller_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_si32_larger_si32_smaller_si32 = arc.cmpi "ne", %larger_si32, %smaller_si32 : si32
    "arc.keep"(%test_ne_si32_larger_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_si32_larger_si32_larger_si32 = arc.cmpi "ne", %larger_si32, %larger_si32 : si32
    "arc.keep"(%test_ne_si32_larger_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_ui64_smaller_ui64_smaller_ui64 = arc.cmpi "ne", %smaller_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_ne_ui64_smaller_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_ui64_smaller_ui64_larger_ui64 = arc.cmpi "ne", %smaller_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_ne_ui64_smaller_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui64_larger_ui64_smaller_ui64 = arc.cmpi "ne", %larger_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_ne_ui64_larger_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_ui64_larger_ui64_larger_ui64 = arc.cmpi "ne", %larger_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_ne_ui64_larger_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_si64_smaller_si64_smaller_si64 = arc.cmpi "ne", %smaller_si64, %smaller_si64 : si64
    "arc.keep"(%test_ne_si64_smaller_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ne_si64_smaller_si64_larger_si64 = arc.cmpi "ne", %smaller_si64, %larger_si64 : si64
    "arc.keep"(%test_ne_si64_smaller_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_si64_larger_si64_smaller_si64 = arc.cmpi "ne", %larger_si64, %smaller_si64 : si64
    "arc.keep"(%test_ne_si64_larger_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ne_si64_larger_si64_larger_si64 = arc.cmpi "ne", %larger_si64, %larger_si64 : si64
    "arc.keep"(%test_ne_si64_larger_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui8_smaller_ui8_smaller_ui8 = arc.cmpi "lt", %smaller_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_lt_ui8_smaller_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui8_smaller_ui8_larger_ui8 = arc.cmpi "lt", %smaller_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_lt_ui8_smaller_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_lt_ui8_larger_ui8_smaller_ui8 = arc.cmpi "lt", %larger_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_lt_ui8_larger_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui8_larger_ui8_larger_ui8 = arc.cmpi "lt", %larger_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_lt_ui8_larger_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si8_smaller_si8_smaller_si8 = arc.cmpi "lt", %smaller_si8, %smaller_si8 : si8
    "arc.keep"(%test_lt_si8_smaller_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si8_smaller_si8_larger_si8 = arc.cmpi "lt", %smaller_si8, %larger_si8 : si8
    "arc.keep"(%test_lt_si8_smaller_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_lt_si8_larger_si8_smaller_si8 = arc.cmpi "lt", %larger_si8, %smaller_si8 : si8
    "arc.keep"(%test_lt_si8_larger_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si8_larger_si8_larger_si8 = arc.cmpi "lt", %larger_si8, %larger_si8 : si8
    "arc.keep"(%test_lt_si8_larger_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui16_smaller_ui16_smaller_ui16 = arc.cmpi "lt", %smaller_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_lt_ui16_smaller_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui16_smaller_ui16_larger_ui16 = arc.cmpi "lt", %smaller_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_lt_ui16_smaller_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_lt_ui16_larger_ui16_smaller_ui16 = arc.cmpi "lt", %larger_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_lt_ui16_larger_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui16_larger_ui16_larger_ui16 = arc.cmpi "lt", %larger_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_lt_ui16_larger_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si16_smaller_si16_smaller_si16 = arc.cmpi "lt", %smaller_si16, %smaller_si16 : si16
    "arc.keep"(%test_lt_si16_smaller_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si16_smaller_si16_larger_si16 = arc.cmpi "lt", %smaller_si16, %larger_si16 : si16
    "arc.keep"(%test_lt_si16_smaller_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_lt_si16_larger_si16_smaller_si16 = arc.cmpi "lt", %larger_si16, %smaller_si16 : si16
    "arc.keep"(%test_lt_si16_larger_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si16_larger_si16_larger_si16 = arc.cmpi "lt", %larger_si16, %larger_si16 : si16
    "arc.keep"(%test_lt_si16_larger_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui32_smaller_ui32_smaller_ui32 = arc.cmpi "lt", %smaller_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_lt_ui32_smaller_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui32_smaller_ui32_larger_ui32 = arc.cmpi "lt", %smaller_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_lt_ui32_smaller_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_lt_ui32_larger_ui32_smaller_ui32 = arc.cmpi "lt", %larger_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_lt_ui32_larger_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui32_larger_ui32_larger_ui32 = arc.cmpi "lt", %larger_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_lt_ui32_larger_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si32_smaller_si32_smaller_si32 = arc.cmpi "lt", %smaller_si32, %smaller_si32 : si32
    "arc.keep"(%test_lt_si32_smaller_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si32_smaller_si32_larger_si32 = arc.cmpi "lt", %smaller_si32, %larger_si32 : si32
    "arc.keep"(%test_lt_si32_smaller_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_lt_si32_larger_si32_smaller_si32 = arc.cmpi "lt", %larger_si32, %smaller_si32 : si32
    "arc.keep"(%test_lt_si32_larger_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si32_larger_si32_larger_si32 = arc.cmpi "lt", %larger_si32, %larger_si32 : si32
    "arc.keep"(%test_lt_si32_larger_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui64_smaller_ui64_smaller_ui64 = arc.cmpi "lt", %smaller_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_lt_ui64_smaller_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui64_smaller_ui64_larger_ui64 = arc.cmpi "lt", %smaller_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_lt_ui64_smaller_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_lt_ui64_larger_ui64_smaller_ui64 = arc.cmpi "lt", %larger_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_lt_ui64_larger_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_ui64_larger_ui64_larger_ui64 = arc.cmpi "lt", %larger_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_lt_ui64_larger_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si64_smaller_si64_smaller_si64 = arc.cmpi "lt", %smaller_si64, %smaller_si64 : si64
    "arc.keep"(%test_lt_si64_smaller_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si64_smaller_si64_larger_si64 = arc.cmpi "lt", %smaller_si64, %larger_si64 : si64
    "arc.keep"(%test_lt_si64_smaller_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_lt_si64_larger_si64_smaller_si64 = arc.cmpi "lt", %larger_si64, %smaller_si64 : si64
    "arc.keep"(%test_lt_si64_larger_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_lt_si64_larger_si64_larger_si64 = arc.cmpi "lt", %larger_si64, %larger_si64 : si64
    "arc.keep"(%test_lt_si64_larger_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui8_smaller_ui8_smaller_ui8 = arc.cmpi "gt", %smaller_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_gt_ui8_smaller_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui8_smaller_ui8_larger_ui8 = arc.cmpi "gt", %smaller_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_gt_ui8_smaller_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui8_larger_ui8_smaller_ui8 = arc.cmpi "gt", %larger_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_gt_ui8_larger_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_gt_ui8_larger_ui8_larger_ui8 = arc.cmpi "gt", %larger_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_gt_ui8_larger_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si8_smaller_si8_smaller_si8 = arc.cmpi "gt", %smaller_si8, %smaller_si8 : si8
    "arc.keep"(%test_gt_si8_smaller_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si8_smaller_si8_larger_si8 = arc.cmpi "gt", %smaller_si8, %larger_si8 : si8
    "arc.keep"(%test_gt_si8_smaller_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si8_larger_si8_smaller_si8 = arc.cmpi "gt", %larger_si8, %smaller_si8 : si8
    "arc.keep"(%test_gt_si8_larger_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_gt_si8_larger_si8_larger_si8 = arc.cmpi "gt", %larger_si8, %larger_si8 : si8
    "arc.keep"(%test_gt_si8_larger_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui16_smaller_ui16_smaller_ui16 = arc.cmpi "gt", %smaller_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_gt_ui16_smaller_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui16_smaller_ui16_larger_ui16 = arc.cmpi "gt", %smaller_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_gt_ui16_smaller_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui16_larger_ui16_smaller_ui16 = arc.cmpi "gt", %larger_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_gt_ui16_larger_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_gt_ui16_larger_ui16_larger_ui16 = arc.cmpi "gt", %larger_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_gt_ui16_larger_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si16_smaller_si16_smaller_si16 = arc.cmpi "gt", %smaller_si16, %smaller_si16 : si16
    "arc.keep"(%test_gt_si16_smaller_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si16_smaller_si16_larger_si16 = arc.cmpi "gt", %smaller_si16, %larger_si16 : si16
    "arc.keep"(%test_gt_si16_smaller_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si16_larger_si16_smaller_si16 = arc.cmpi "gt", %larger_si16, %smaller_si16 : si16
    "arc.keep"(%test_gt_si16_larger_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_gt_si16_larger_si16_larger_si16 = arc.cmpi "gt", %larger_si16, %larger_si16 : si16
    "arc.keep"(%test_gt_si16_larger_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui32_smaller_ui32_smaller_ui32 = arc.cmpi "gt", %smaller_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_gt_ui32_smaller_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui32_smaller_ui32_larger_ui32 = arc.cmpi "gt", %smaller_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_gt_ui32_smaller_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui32_larger_ui32_smaller_ui32 = arc.cmpi "gt", %larger_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_gt_ui32_larger_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_gt_ui32_larger_ui32_larger_ui32 = arc.cmpi "gt", %larger_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_gt_ui32_larger_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si32_smaller_si32_smaller_si32 = arc.cmpi "gt", %smaller_si32, %smaller_si32 : si32
    "arc.keep"(%test_gt_si32_smaller_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si32_smaller_si32_larger_si32 = arc.cmpi "gt", %smaller_si32, %larger_si32 : si32
    "arc.keep"(%test_gt_si32_smaller_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si32_larger_si32_smaller_si32 = arc.cmpi "gt", %larger_si32, %smaller_si32 : si32
    "arc.keep"(%test_gt_si32_larger_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_gt_si32_larger_si32_larger_si32 = arc.cmpi "gt", %larger_si32, %larger_si32 : si32
    "arc.keep"(%test_gt_si32_larger_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui64_smaller_ui64_smaller_ui64 = arc.cmpi "gt", %smaller_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_gt_ui64_smaller_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui64_smaller_ui64_larger_ui64 = arc.cmpi "gt", %smaller_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_gt_ui64_smaller_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_ui64_larger_ui64_smaller_ui64 = arc.cmpi "gt", %larger_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_gt_ui64_larger_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_gt_ui64_larger_ui64_larger_ui64 = arc.cmpi "gt", %larger_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_gt_ui64_larger_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si64_smaller_si64_smaller_si64 = arc.cmpi "gt", %smaller_si64, %smaller_si64 : si64
    "arc.keep"(%test_gt_si64_smaller_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si64_smaller_si64_larger_si64 = arc.cmpi "gt", %smaller_si64, %larger_si64 : si64
    "arc.keep"(%test_gt_si64_smaller_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_gt_si64_larger_si64_smaller_si64 = arc.cmpi "gt", %larger_si64, %smaller_si64 : si64
    "arc.keep"(%test_gt_si64_larger_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_gt_si64_larger_si64_larger_si64 = arc.cmpi "gt", %larger_si64, %larger_si64 : si64
    "arc.keep"(%test_gt_si64_larger_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_ui8_smaller_ui8_smaller_ui8 = arc.cmpi "ge", %smaller_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_ge_ui8_smaller_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui8_smaller_ui8_larger_ui8 = arc.cmpi "ge", %smaller_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_ge_ui8_smaller_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_ui8_larger_ui8_smaller_ui8 = arc.cmpi "ge", %larger_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_ge_ui8_larger_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui8_larger_ui8_larger_ui8 = arc.cmpi "ge", %larger_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_ge_ui8_larger_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si8_smaller_si8_smaller_si8 = arc.cmpi "ge", %smaller_si8, %smaller_si8 : si8
    "arc.keep"(%test_ge_si8_smaller_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si8_smaller_si8_larger_si8 = arc.cmpi "ge", %smaller_si8, %larger_si8 : si8
    "arc.keep"(%test_ge_si8_smaller_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_si8_larger_si8_smaller_si8 = arc.cmpi "ge", %larger_si8, %smaller_si8 : si8
    "arc.keep"(%test_ge_si8_larger_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si8_larger_si8_larger_si8 = arc.cmpi "ge", %larger_si8, %larger_si8 : si8
    "arc.keep"(%test_ge_si8_larger_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui16_smaller_ui16_smaller_ui16 = arc.cmpi "ge", %smaller_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_ge_ui16_smaller_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui16_smaller_ui16_larger_ui16 = arc.cmpi "ge", %smaller_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_ge_ui16_smaller_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_ui16_larger_ui16_smaller_ui16 = arc.cmpi "ge", %larger_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_ge_ui16_larger_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui16_larger_ui16_larger_ui16 = arc.cmpi "ge", %larger_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_ge_ui16_larger_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si16_smaller_si16_smaller_si16 = arc.cmpi "ge", %smaller_si16, %smaller_si16 : si16
    "arc.keep"(%test_ge_si16_smaller_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si16_smaller_si16_larger_si16 = arc.cmpi "ge", %smaller_si16, %larger_si16 : si16
    "arc.keep"(%test_ge_si16_smaller_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_si16_larger_si16_smaller_si16 = arc.cmpi "ge", %larger_si16, %smaller_si16 : si16
    "arc.keep"(%test_ge_si16_larger_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si16_larger_si16_larger_si16 = arc.cmpi "ge", %larger_si16, %larger_si16 : si16
    "arc.keep"(%test_ge_si16_larger_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui32_smaller_ui32_smaller_ui32 = arc.cmpi "ge", %smaller_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_ge_ui32_smaller_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui32_smaller_ui32_larger_ui32 = arc.cmpi "ge", %smaller_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_ge_ui32_smaller_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_ui32_larger_ui32_smaller_ui32 = arc.cmpi "ge", %larger_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_ge_ui32_larger_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui32_larger_ui32_larger_ui32 = arc.cmpi "ge", %larger_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_ge_ui32_larger_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si32_smaller_si32_smaller_si32 = arc.cmpi "ge", %smaller_si32, %smaller_si32 : si32
    "arc.keep"(%test_ge_si32_smaller_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si32_smaller_si32_larger_si32 = arc.cmpi "ge", %smaller_si32, %larger_si32 : si32
    "arc.keep"(%test_ge_si32_smaller_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_si32_larger_si32_smaller_si32 = arc.cmpi "ge", %larger_si32, %smaller_si32 : si32
    "arc.keep"(%test_ge_si32_larger_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si32_larger_si32_larger_si32 = arc.cmpi "ge", %larger_si32, %larger_si32 : si32
    "arc.keep"(%test_ge_si32_larger_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui64_smaller_ui64_smaller_ui64 = arc.cmpi "ge", %smaller_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_ge_ui64_smaller_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui64_smaller_ui64_larger_ui64 = arc.cmpi "ge", %smaller_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_ge_ui64_smaller_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_ui64_larger_ui64_smaller_ui64 = arc.cmpi "ge", %larger_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_ge_ui64_larger_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_ui64_larger_ui64_larger_ui64 = arc.cmpi "ge", %larger_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_ge_ui64_larger_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si64_smaller_si64_smaller_si64 = arc.cmpi "ge", %smaller_si64, %smaller_si64 : si64
    "arc.keep"(%test_ge_si64_smaller_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si64_smaller_si64_larger_si64 = arc.cmpi "ge", %smaller_si64, %larger_si64 : si64
    "arc.keep"(%test_ge_si64_smaller_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_ge_si64_larger_si64_smaller_si64 = arc.cmpi "ge", %larger_si64, %smaller_si64 : si64
    "arc.keep"(%test_ge_si64_larger_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_ge_si64_larger_si64_larger_si64 = arc.cmpi "ge", %larger_si64, %larger_si64 : si64
    "arc.keep"(%test_ge_si64_larger_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui8_smaller_ui8_smaller_ui8 = arc.cmpi "le", %smaller_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_le_ui8_smaller_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui8_smaller_ui8_larger_ui8 = arc.cmpi "le", %smaller_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_le_ui8_smaller_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui8_larger_ui8_smaller_ui8 = arc.cmpi "le", %larger_ui8, %smaller_ui8 : ui8
    "arc.keep"(%test_le_ui8_larger_ui8_smaller_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_le_ui8_larger_ui8_larger_ui8 = arc.cmpi "le", %larger_ui8, %larger_ui8 : ui8
    "arc.keep"(%test_le_ui8_larger_ui8_larger_ui8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si8_smaller_si8_smaller_si8 = arc.cmpi "le", %smaller_si8, %smaller_si8 : si8
    "arc.keep"(%test_le_si8_smaller_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si8_smaller_si8_larger_si8 = arc.cmpi "le", %smaller_si8, %larger_si8 : si8
    "arc.keep"(%test_le_si8_smaller_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si8_larger_si8_smaller_si8 = arc.cmpi "le", %larger_si8, %smaller_si8 : si8
    "arc.keep"(%test_le_si8_larger_si8_smaller_si8) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_le_si8_larger_si8_larger_si8 = arc.cmpi "le", %larger_si8, %larger_si8 : si8
    "arc.keep"(%test_le_si8_larger_si8_larger_si8) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui16_smaller_ui16_smaller_ui16 = arc.cmpi "le", %smaller_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_le_ui16_smaller_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui16_smaller_ui16_larger_ui16 = arc.cmpi "le", %smaller_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_le_ui16_smaller_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui16_larger_ui16_smaller_ui16 = arc.cmpi "le", %larger_ui16, %smaller_ui16 : ui16
    "arc.keep"(%test_le_ui16_larger_ui16_smaller_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_le_ui16_larger_ui16_larger_ui16 = arc.cmpi "le", %larger_ui16, %larger_ui16 : ui16
    "arc.keep"(%test_le_ui16_larger_ui16_larger_ui16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si16_smaller_si16_smaller_si16 = arc.cmpi "le", %smaller_si16, %smaller_si16 : si16
    "arc.keep"(%test_le_si16_smaller_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si16_smaller_si16_larger_si16 = arc.cmpi "le", %smaller_si16, %larger_si16 : si16
    "arc.keep"(%test_le_si16_smaller_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si16_larger_si16_smaller_si16 = arc.cmpi "le", %larger_si16, %smaller_si16 : si16
    "arc.keep"(%test_le_si16_larger_si16_smaller_si16) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_le_si16_larger_si16_larger_si16 = arc.cmpi "le", %larger_si16, %larger_si16 : si16
    "arc.keep"(%test_le_si16_larger_si16_larger_si16) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui32_smaller_ui32_smaller_ui32 = arc.cmpi "le", %smaller_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_le_ui32_smaller_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui32_smaller_ui32_larger_ui32 = arc.cmpi "le", %smaller_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_le_ui32_smaller_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui32_larger_ui32_smaller_ui32 = arc.cmpi "le", %larger_ui32, %smaller_ui32 : ui32
    "arc.keep"(%test_le_ui32_larger_ui32_smaller_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_le_ui32_larger_ui32_larger_ui32 = arc.cmpi "le", %larger_ui32, %larger_ui32 : ui32
    "arc.keep"(%test_le_ui32_larger_ui32_larger_ui32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si32_smaller_si32_smaller_si32 = arc.cmpi "le", %smaller_si32, %smaller_si32 : si32
    "arc.keep"(%test_le_si32_smaller_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si32_smaller_si32_larger_si32 = arc.cmpi "le", %smaller_si32, %larger_si32 : si32
    "arc.keep"(%test_le_si32_smaller_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si32_larger_si32_smaller_si32 = arc.cmpi "le", %larger_si32, %smaller_si32 : si32
    "arc.keep"(%test_le_si32_larger_si32_smaller_si32) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_le_si32_larger_si32_larger_si32 = arc.cmpi "le", %larger_si32, %larger_si32 : si32
    "arc.keep"(%test_le_si32_larger_si32_larger_si32) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui64_smaller_ui64_smaller_ui64 = arc.cmpi "le", %smaller_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_le_ui64_smaller_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui64_smaller_ui64_larger_ui64 = arc.cmpi "le", %smaller_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_le_ui64_smaller_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_ui64_larger_ui64_smaller_ui64 = arc.cmpi "le", %larger_ui64, %smaller_ui64 : ui64
    "arc.keep"(%test_le_ui64_larger_ui64_smaller_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_le_ui64_larger_ui64_larger_ui64 = arc.cmpi "le", %larger_ui64, %larger_ui64 : ui64
    "arc.keep"(%test_le_ui64_larger_ui64_larger_ui64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si64_smaller_si64_smaller_si64 = arc.cmpi "le", %smaller_si64, %smaller_si64 : si64
    "arc.keep"(%test_le_si64_smaller_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si64_smaller_si64_larger_si64 = arc.cmpi "le", %smaller_si64, %larger_si64 : si64
    "arc.keep"(%test_le_si64_smaller_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

    %test_le_si64_larger_si64_smaller_si64 = arc.cmpi "le", %larger_si64, %smaller_si64 : si64
    "arc.keep"(%test_le_si64_larger_si64_smaller_si64) : (i1) -> ()
// CHECK: "arc.keep"([[FALSE]]) : (i1) -> ()

    %test_le_si64_larger_si64_larger_si64 = arc.cmpi "le", %larger_si64, %larger_si64 : si64
    "arc.keep"(%test_le_si64_larger_si64_larger_si64) : (i1) -> ()
// CHECK: "arc.keep"([[TRUE]]) : (i1) -> ()

// Check that we don't fold anything that's not foldable
    %test_nofold = arc.cmpi "le", %arg0, %larger_ui64 : ui64
// CHECK-DAG: [[NOFOLD:%[^ ]+]] = arc.cmpi le, %arg0, [[LARGERU64]] : ui64
    "arc.keep"(%test_nofold) : (i1) -> ()
// CHECK: "arc.keep"([[NOFOLD]]) : (i1) -> ()

    return
  }
}
