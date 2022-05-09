// RUN: arc-mlir --canonicalize %s | FileCheck %s
module @toplevel {
  func.func @main(%arg0 : i64) {
    %si8_max = arc.constant 127 : si8
    %si8_max_minus_1 = arc.constant 126 : si8
    %si8_one = arc.constant 1 : si8
    %ui8_max = arc.constant 255 : ui8
    %ui8_max_minus_1 = arc.constant 254 : ui8
    %ui8_one = arc.constant 1 : ui8
    // CHECK-DAG: [[si8MAX:%[^ ]+]] = arc.constant 127 : si8
    // CHECK-DAG: [[si8MAX1:%[^ ]+]] = arc.constant 126 : si8
    // CHECK-DAG: [[ui8MAX:%[^ ]+]] = arc.constant 255 : ui8
    // CHECK-DAG: [[ui8MAX1:%[^ ]+]] = arc.constant 254 : ui8
    %si16_max = arc.constant 32767 : si16
    %si16_max_minus_1 = arc.constant 32766 : si16
    %si16_one = arc.constant 1 : si16
    %ui16_max = arc.constant 65535 : ui16
    %ui16_max_minus_1 = arc.constant 65534 : ui16
    %ui16_one = arc.constant 1 : ui16
    // CHECK-DAG: [[si16MAX:%[^ ]+]] = arc.constant 32767 : si16
    // CHECK-DAG: [[si16MAX1:%[^ ]+]] = arc.constant 32766 : si16
    // CHECK-DAG: [[ui16MAX:%[^ ]+]] = arc.constant 65535 : ui16
    // CHECK-DAG: [[ui16MAX1:%[^ ]+]] = arc.constant 65534 : ui16
    %si32_max = arc.constant 2147483647 : si32
    %si32_max_minus_1 = arc.constant 2147483646 : si32
    %si32_one = arc.constant 1 : si32
    %ui32_max = arc.constant 4294967295 : ui32
    %ui32_max_minus_1 = arc.constant 4294967294 : ui32
    %ui32_one = arc.constant 1 : ui32
    // CHECK-DAG: [[si32MAX:%[^ ]+]] = arc.constant 2147483647 : si32
    // CHECK-DAG: [[si32MAX1:%[^ ]+]] = arc.constant 2147483646 : si32
    // CHECK-DAG: [[ui32MAX:%[^ ]+]] = arc.constant 4294967295 : ui32
    // CHECK-DAG: [[ui32MAX1:%[^ ]+]] = arc.constant 4294967294 : ui32
    %si64_max = arc.constant 9223372036854775807 : si64
    %si64_max_minus_1 = arc.constant 9223372036854775806 : si64
    %si64_one = arc.constant 1 : si64
    %ui64_max = arc.constant 18446744073709551615 : ui64
    %ui64_max_minus_1 = arc.constant 18446744073709551614 : ui64
    %ui64_one = arc.constant 1 : ui64
    // CHECK-DAG: [[si64MAX:%[^ ]+]] = arc.constant 9223372036854775807 : si64
    // CHECK-DAG: [[si64MAX1:%[^ ]+]] = arc.constant 9223372036854775806 : si64
    // CHECK-DAG: [[ui64MAX:%[^ ]+]] = arc.constant 18446744073709551615 : ui64
    // CHECK-DAG: [[ui64MAX1:%[^ ]+]] = arc.constant 18446744073709551614 : ui64
    // Check that we don't fold something that would overflow
    %si8_nofold = arc.addi %si8_max, %si8_max_minus_1 : si8
    // CHECK-DAG: [[si8NOFOLD:%[^ ]+]] = arc.addi [[si8MAX]], [[si8MAX1]] : si8
    // CHECK: "arc.keep"([[si8NOFOLD]]) : (si8) -> ()
    "arc.keep"(%si8_nofold) : (si8) -> ()
    // Check that we fold when we stay in range
    %si8_fold = arc.addi %si8_one, %si8_max_minus_1 : si8
    // CHECK: "arc.keep"([[si8MAX]]) : (si8) -> ()
    "arc.keep"(%si8_fold) : (si8) -> ()
    // Check that we don't fold something that would overflow
    %ui8_nofold = arc.addi %ui8_max, %ui8_max_minus_1 : ui8
    // CHECK-DAG: [[ui8NOFOLD:%[^ ]+]] = arc.addi [[ui8MAX]], [[ui8MAX1]] : ui8
    // CHECK: "arc.keep"([[ui8NOFOLD]]) : (ui8) -> ()
    "arc.keep"(%ui8_nofold) : (ui8) -> ()
    // Check that we fold when we stay in range
    %ui8_fold = arc.addi %ui8_one, %ui8_max_minus_1 : ui8
    // CHECK: "arc.keep"([[ui8MAX]]) : (ui8) -> ()
    "arc.keep"(%ui8_fold) : (ui8) -> ()
    // Check that we don't fold something that would overflow
    %si16_nofold = arc.addi %si16_max, %si16_max_minus_1 : si16
    // CHECK-DAG: [[si16NOFOLD:%[^ ]+]] = arc.addi [[si16MAX]], [[si16MAX1]] : si16
    // CHECK: "arc.keep"([[si16NOFOLD]]) : (si16) -> ()
    "arc.keep"(%si16_nofold) : (si16) -> ()
    // Check that we fold when we stay in range
    %si16_fold = arc.addi %si16_one, %si16_max_minus_1 : si16
    // CHECK: "arc.keep"([[si16MAX]]) : (si16) -> ()
    "arc.keep"(%si16_fold) : (si16) -> ()
    // Check that we don't fold something that would overflow
    %ui16_nofold = arc.addi %ui16_max, %ui16_max_minus_1 : ui16
    // CHECK-DAG: [[ui16NOFOLD:%[^ ]+]] = arc.addi [[ui16MAX]], [[ui16MAX1]] : ui16
    // CHECK: "arc.keep"([[ui16NOFOLD]]) : (ui16) -> ()
    "arc.keep"(%ui16_nofold) : (ui16) -> ()
    // Check that we fold when we stay in range
    %ui16_fold = arc.addi %ui16_one, %ui16_max_minus_1 : ui16
    // CHECK: "arc.keep"([[ui16MAX]]) : (ui16) -> ()
    "arc.keep"(%ui16_fold) : (ui16) -> ()
    // Check that we don't fold something that would overflow
    %si32_nofold = arc.addi %si32_max, %si32_max_minus_1 : si32
    // CHECK-DAG: [[si32NOFOLD:%[^ ]+]] = arc.addi [[si32MAX]], [[si32MAX1]] : si32
    // CHECK: "arc.keep"([[si32NOFOLD]]) : (si32) -> ()
    "arc.keep"(%si32_nofold) : (si32) -> ()
    // Check that we fold when we stay in range
    %si32_fold = arc.addi %si32_one, %si32_max_minus_1 : si32
    // CHECK: "arc.keep"([[si32MAX]]) : (si32) -> ()
    "arc.keep"(%si32_fold) : (si32) -> ()
    // Check that we don't fold something that would overflow
    %ui32_nofold = arc.addi %ui32_max, %ui32_max_minus_1 : ui32
    // CHECK-DAG: [[ui32NOFOLD:%[^ ]+]] = arc.addi [[ui32MAX]], [[ui32MAX1]] : ui32
    // CHECK: "arc.keep"([[ui32NOFOLD]]) : (ui32) -> ()
    "arc.keep"(%ui32_nofold) : (ui32) -> ()
    // Check that we fold when we stay in range
    %ui32_fold = arc.addi %ui32_one, %ui32_max_minus_1 : ui32
    // CHECK: "arc.keep"([[ui32MAX]]) : (ui32) -> ()
    "arc.keep"(%ui32_fold) : (ui32) -> ()
    // Check that we don't fold something that would overflow
    %si64_nofold = arc.addi %si64_max, %si64_max_minus_1 : si64
    // CHECK-DAG: [[si64NOFOLD:%[^ ]+]] = arc.addi [[si64MAX]], [[si64MAX1]] : si64
    // CHECK: "arc.keep"([[si64NOFOLD]]) : (si64) -> ()
    "arc.keep"(%si64_nofold) : (si64) -> ()
    // Check that we fold when we stay in range
    %si64_fold = arc.addi %si64_one, %si64_max_minus_1 : si64
    // CHECK: "arc.keep"([[si64MAX]]) : (si64) -> ()
    "arc.keep"(%si64_fold) : (si64) -> ()
    // Check that we don't fold something that would overflow
    %ui64_nofold = arc.addi %ui64_max, %ui64_max_minus_1 : ui64
    // CHECK-DAG: [[ui64NOFOLD:%[^ ]+]] = arc.addi [[ui64MAX]], [[ui64MAX1]] : ui64
    // CHECK: "arc.keep"([[ui64NOFOLD]]) : (ui64) -> ()
    "arc.keep"(%ui64_nofold) : (ui64) -> ()
    // Check that we fold when we stay in range
    %ui64_fold = arc.addi %ui64_one, %ui64_max_minus_1 : ui64
    // CHECK: "arc.keep"([[ui64MAX]]) : (ui64) -> ()
    "arc.keep"(%ui64_fold) : (ui64) -> ()
    return
  }
}
