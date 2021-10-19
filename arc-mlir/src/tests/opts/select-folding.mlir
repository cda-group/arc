// RUN: arc-mlir --canonicalize %s | FileCheck %s
module @toplevel {
  func @main(%arg0 : i1) {
    %true = arith.constant 1 : i1
    %false = arith.constant 0 : i1

    %si8_left = arc.constant 42 : si8
    %si8_right = arc.constant 17 : si8
    %ui8_left = arc.constant 43 : ui8
    %ui8_right = arc.constant 18 : ui8
    // CHECK-DAG: [[LEFT_si8:%[^ ]+]] = arc.constant 42 : si8
    // CHECK-DAG: [[RIGHT_si8:%[^ ]+]] = arc.constant 17 : si8
    // CHECK-DAG: [[LEFT_ui8:%[^ ]+]] = arc.constant 43 : ui8
    // CHECK-DAG: [[RIGHT_ui8:%[^ ]+]] = arc.constant 18 : ui8
    %si16_left = arc.constant 42 : si16
    %si16_right = arc.constant 17 : si16
    %ui16_left = arc.constant 43 : ui16
    %ui16_right = arc.constant 18 : ui16
    // CHECK-DAG: [[LEFT_si16:%[^ ]+]] = arc.constant 42 : si16
    // CHECK-DAG: [[RIGHT_si16:%[^ ]+]] = arc.constant 17 : si16
    // CHECK-DAG: [[LEFT_ui16:%[^ ]+]] = arc.constant 43 : ui16
    // CHECK-DAG: [[RIGHT_ui16:%[^ ]+]] = arc.constant 18 : ui16
    %si32_left = arc.constant 42 : si32
    %si32_right = arc.constant 17 : si32
    %ui32_left = arc.constant 43 : ui32
    %ui32_right = arc.constant 18 : ui32
    // CHECK-DAG: [[LEFT_si32:%[^ ]+]] = arc.constant 42 : si32
    // CHECK-DAG: [[RIGHT_si32:%[^ ]+]] = arc.constant 17 : si32
    // CHECK-DAG: [[LEFT_ui32:%[^ ]+]] = arc.constant 43 : ui32
    // CHECK-DAG: [[RIGHT_ui32:%[^ ]+]] = arc.constant 18 : ui32
    %si64_left = arc.constant 42 : si64
    %si64_right = arc.constant 17 : si64
    %ui64_left = arc.constant 43 : ui64
    %ui64_right = arc.constant 18 : ui64
    // CHECK-DAG: [[LEFT_si64:%[^ ]+]] = arc.constant 42 : si64
    // CHECK-DAG: [[RIGHT_si64:%[^ ]+]] = arc.constant 17 : si64
    // CHECK-DAG: [[LEFT_ui64:%[^ ]+]] = arc.constant 43 : ui64
    // CHECK-DAG: [[RIGHT_ui64:%[^ ]+]] = arc.constant 18 : ui64

    %result_ui8_true = arc.select %true, %ui8_left, %ui8_right : ui8
    "arc.keep"(%result_ui8_true) : (ui8) -> ()
    // CHECK: "arc.keep"([[LEFT_ui8]]) : (ui8) -> ()


    %result_ui8_false = arc.select %false, %ui8_left, %ui8_right : ui8
    "arc.keep"(%result_ui8_false) : (ui8) -> ()
    // CHECK: "arc.keep"([[RIGHT_ui8]]) : (ui8) -> ()


    %result_ui8_unknown = arc.select %arg0, %ui8_left, %ui8_right : ui8
    "arc.keep"(%result_ui8_unknown) : (ui8) -> ()
    // CHECK: [[UNKNOWN_ui8:%[^ ]+]] = arc.select %arg0, [[LEFT_ui8]], [[RIGHT_ui8]]
    // CHECK: "arc.keep"([[UNKNOWN_ui8]]) : (ui8) -> ()


    %result_si8_true = arc.select %true, %si8_left, %si8_right : si8
    "arc.keep"(%result_si8_true) : (si8) -> ()
    // CHECK: "arc.keep"([[LEFT_si8]]) : (si8) -> ()


    %result_si8_false = arc.select %false, %si8_left, %si8_right : si8
    "arc.keep"(%result_si8_false) : (si8) -> ()
    // CHECK: "arc.keep"([[RIGHT_si8]]) : (si8) -> ()


    %result_si8_unknown = arc.select %arg0, %si8_left, %si8_right : si8
    "arc.keep"(%result_si8_unknown) : (si8) -> ()
    // CHECK: [[UNKNOWN_si8:%[^ ]+]] = arc.select %arg0, [[LEFT_si8]], [[RIGHT_si8]]
    // CHECK: "arc.keep"([[UNKNOWN_si8]]) : (si8) -> ()


    %result_ui16_true = arc.select %true, %ui16_left, %ui16_right : ui16
    "arc.keep"(%result_ui16_true) : (ui16) -> ()
    // CHECK: "arc.keep"([[LEFT_ui16]]) : (ui16) -> ()


    %result_ui16_false = arc.select %false, %ui16_left, %ui16_right : ui16
    "arc.keep"(%result_ui16_false) : (ui16) -> ()
    // CHECK: "arc.keep"([[RIGHT_ui16]]) : (ui16) -> ()


    %result_ui16_unknown = arc.select %arg0, %ui16_left, %ui16_right : ui16
    "arc.keep"(%result_ui16_unknown) : (ui16) -> ()
    // CHECK: [[UNKNOWN_ui16:%[^ ]+]] = arc.select %arg0, [[LEFT_ui16]], [[RIGHT_ui16]]
    // CHECK: "arc.keep"([[UNKNOWN_ui16]]) : (ui16) -> ()


    %result_si16_true = arc.select %true, %si16_left, %si16_right : si16
    "arc.keep"(%result_si16_true) : (si16) -> ()
    // CHECK: "arc.keep"([[LEFT_si16]]) : (si16) -> ()


    %result_si16_false = arc.select %false, %si16_left, %si16_right : si16
    "arc.keep"(%result_si16_false) : (si16) -> ()
    // CHECK: "arc.keep"([[RIGHT_si16]]) : (si16) -> ()


    %result_si16_unknown = arc.select %arg0, %si16_left, %si16_right : si16
    "arc.keep"(%result_si16_unknown) : (si16) -> ()
    // CHECK: [[UNKNOWN_si16:%[^ ]+]] = arc.select %arg0, [[LEFT_si16]], [[RIGHT_si16]]
    // CHECK: "arc.keep"([[UNKNOWN_si16]]) : (si16) -> ()


    %result_ui32_true = arc.select %true, %ui32_left, %ui32_right : ui32
    "arc.keep"(%result_ui32_true) : (ui32) -> ()
    // CHECK: "arc.keep"([[LEFT_ui32]]) : (ui32) -> ()


    %result_ui32_false = arc.select %false, %ui32_left, %ui32_right : ui32
    "arc.keep"(%result_ui32_false) : (ui32) -> ()
    // CHECK: "arc.keep"([[RIGHT_ui32]]) : (ui32) -> ()


    %result_ui32_unknown = arc.select %arg0, %ui32_left, %ui32_right : ui32
    "arc.keep"(%result_ui32_unknown) : (ui32) -> ()
    // CHECK: [[UNKNOWN_ui32:%[^ ]+]] = arc.select %arg0, [[LEFT_ui32]], [[RIGHT_ui32]]
    // CHECK: "arc.keep"([[UNKNOWN_ui32]]) : (ui32) -> ()


    %result_si32_true = arc.select %true, %si32_left, %si32_right : si32
    "arc.keep"(%result_si32_true) : (si32) -> ()
    // CHECK: "arc.keep"([[LEFT_si32]]) : (si32) -> ()


    %result_si32_false = arc.select %false, %si32_left, %si32_right : si32
    "arc.keep"(%result_si32_false) : (si32) -> ()
    // CHECK: "arc.keep"([[RIGHT_si32]]) : (si32) -> ()


    %result_si32_unknown = arc.select %arg0, %si32_left, %si32_right : si32
    "arc.keep"(%result_si32_unknown) : (si32) -> ()
    // CHECK: [[UNKNOWN_si32:%[^ ]+]] = arc.select %arg0, [[LEFT_si32]], [[RIGHT_si32]]
    // CHECK: "arc.keep"([[UNKNOWN_si32]]) : (si32) -> ()


    %result_ui64_true = arc.select %true, %ui64_left, %ui64_right : ui64
    "arc.keep"(%result_ui64_true) : (ui64) -> ()
    // CHECK: "arc.keep"([[LEFT_ui64]]) : (ui64) -> ()


    %result_ui64_false = arc.select %false, %ui64_left, %ui64_right : ui64
    "arc.keep"(%result_ui64_false) : (ui64) -> ()
    // CHECK: "arc.keep"([[RIGHT_ui64]]) : (ui64) -> ()


    %result_ui64_unknown = arc.select %arg0, %ui64_left, %ui64_right : ui64
    "arc.keep"(%result_ui64_unknown) : (ui64) -> ()
    // CHECK: [[UNKNOWN_ui64:%[^ ]+]] = arc.select %arg0, [[LEFT_ui64]], [[RIGHT_ui64]]
    // CHECK: "arc.keep"([[UNKNOWN_ui64]]) : (ui64) -> ()


    %result_si64_true = arc.select %true, %si64_left, %si64_right : si64
    "arc.keep"(%result_si64_true) : (si64) -> ()
    // CHECK: "arc.keep"([[LEFT_si64]]) : (si64) -> ()


    %result_si64_false = arc.select %false, %si64_left, %si64_right : si64
    "arc.keep"(%result_si64_false) : (si64) -> ()
    // CHECK: "arc.keep"([[RIGHT_si64]]) : (si64) -> ()


    %result_si64_unknown = arc.select %arg0, %si64_left, %si64_right : si64
    "arc.keep"(%result_si64_unknown) : (si64) -> ()
    // CHECK: [[UNKNOWN_si64:%[^ ]+]] = arc.select %arg0, [[LEFT_si64]], [[RIGHT_si64]]
    // CHECK: "arc.keep"([[UNKNOWN_si64]]) : (si64) -> ()


    return
  }
}
