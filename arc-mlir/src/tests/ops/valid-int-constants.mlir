// Check that we can parse our own integer constants and that we can
// round-trip over the generic operator format.
//
// RUN: arc-mlir %s | FileCheck %s
// RUN: arc-mlir -mlir-print-op-generic %s | arc-mlir | FileCheck %s
module @toplevel {
  func @main() {
    %si8_min = arc.constant -128 : si8
//CHECK: {{%[^ ]+}} = arc.constant -128 : si8
    %si8_max = arc.constant 127 : si8
//CHECK: {{%[^ ]+}} = arc.constant 127 : si8
    %si16_min = arc.constant -32768 : si16
//CHECK: {{%[^ ]+}} = arc.constant -32768 : si16
    %si16_max = arc.constant 32767 : si16
//CHECK: {{%[^ ]+}} = arc.constant 32767 : si16
    %si32_min = arc.constant -2147483648 : si32
//CHECK: {{%[^ ]+}} = arc.constant -2147483648 : si32
    %si32_max = arc.constant 2147483647 : si32
//CHECK: {{%[^ ]+}} = arc.constant 2147483647 : si32
    %si64_min = arc.constant -9223372036854775808 : si64
//CHECK: {{%[^ ]+}} = arc.constant -9223372036854775808 : si64
    %si64_max = arc.constant 9223372036854775807 : si64
//CHECK: {{%[^ ]+}} = arc.constant 9223372036854775807 : si64
    %ui8_max = arc.constant 255 : ui8
//CHECK: {{%[^ ]+}} = arc.constant 255 : ui8
    %ui16_max = arc.constant 65535 : ui16
//CHECK: {{%[^ ]+}} = arc.constant 65535 : ui16
    %ui32_max = arc.constant 4294967295 : ui32
//CHECK: {{%[^ ]+}} = arc.constant 4294967295 : ui32
    %ui64_max = arc.constant 18446744073709551615 : ui64
//CHECK: {{%[^ ]+}} = arc.constant 18446744073709551615 : ui64
    return
  }
}
