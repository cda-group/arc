// RUN: arc-mlir -split-input-file --canonicalize %s | FileCheck %s

module @toplevel {
  func @main() -> i1 {
    %a = constant 0 : i1
    %b = constant 1 : i1

    %tuple = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>
    %elem = "arc.index_tuple"(%tuple) { index = 0 } : (tuple<i1,i1>) -> i1

    return %elem : i1
    // CHECK-DAG: [[FALSE:%[^ ]+]] = constant 0 : i1
    // CHECK: return [[FALSE]] : i1
  }
}

// -----

module @toplevel {
  func @main() -> si32 {
    %a = arc.constant 7 : si32
    %b = arc.constant 17 : si32

    %tuple = "arc.make_tuple"(%a, %b) : (si32, si32) -> tuple<si32,si32>

    %elem = "arc.index_tuple"(%tuple) { index = 1 } : (tuple<si32,si32>) -> si32

    return %elem : si32
    // CHECK-DAG: [[SEVENTEEN:%[^ ]+]] = arc.constant 17 : si32
    // CHECK: return [[SEVENTEEN]] : si32
  }
}

// -----

module @toplevel {
  func @main(%tuple : tuple<si32,si32>) -> si32 {
    %elem = "arc.index_tuple"(%tuple) { index = 1 } : (tuple<si32,si32>) -> si32

    return %elem : si32
    // CHECK-DAG: [[INDEXTUPLE:%[^ ]+]] = "arc.index_tuple"
    // CHECK: return [[INDEXTUPLE]] : si32
  }
}

