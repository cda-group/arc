// RUN: arc-mlir-rust-test %t %s

module @"loop_crate" {

"rust.func"() ( {
  ^bb0(%arg0: !rust<"u64">, %arg1: !rust<"u64">, %arg2: !rust<"u64">):
    %0:2 = "rust.loop"(%arg0, %arg2) ( {
    ^bb0(%arg3: !rust<"u64">, %arg4: !rust<"u64">):  // no predecessors
      %1 = "rust.compop"(%arg3, %arg1) {op = "<"} : (!rust<"u64">, !rust<"u64">) -> !rust<"bool">
      "rust.loop.condition"(%1, %arg3, %arg4) : (!rust<"bool">, !rust<"u64">, !rust<"u64">) -> ()
    },  {
    ^bb0(%arg3: !rust<"u64">, %arg4: !rust<"u64">):  // no predecessors
      %1 = "rust.constant"() {value = "1" : !rust<"u64">} : () -> !rust<"u64">
      %2 = "rust.binaryop"(%arg3, %1) {op="+"} : (!rust<"u64">, !rust<"u64">) -> !rust<"u64">
      %3 = "rust.binaryop"(%arg3, %arg4) {op="+"} : (!rust<"u64">, !rust<"u64">) -> !rust<"u64">
      "rust.loop.yield"(%2, %3) : (!rust<"u64">, !rust<"u64">) -> ()
    }) : (!rust<"u64">, !rust<"u64">) -> (!rust<"u64">, !rust<"u64">)
    "rust.return"(%0#1) : (!rust<"u64">) -> ()
}) {
    sym_name = "a_while_loop",
    type = (!rust<"u64">, !rust<"u64">, !rust<"u64">) -> !rust<"u64">
   } : () -> ()
}

