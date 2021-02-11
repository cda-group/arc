// RUN: arc-mlir -split-input-file --canonicalize %s | FileCheck %s

module @toplevel {
  func @main() -> i1 {
    %a = constant 0 : i1
    %b = constant 1 : i1

    %struct = arc.make_struct(%a, %b : i1, i1) : !arc.struct<a : i1, b : i1>
    %elem = "arc.struct_access"(%struct) { field = "a" } : (!arc.struct<a : i1, b : i1>) -> i1

    return %elem : i1
    // CHECK-DAG: [[FALSE:%[^ ]+]] = constant false
    // CHECK: return [[FALSE]] : i1
  }
}

// -----

module @toplevel {
  func @main() -> si32 {
    %a = arc.constant 7 : si32
    %b = arc.constant 17 : si32

    %struct = arc.make_struct(%a, %b : si32, si32) : !arc.struct<a : si32, b : si32>

    %elem = "arc.struct_access"(%struct) { field = "b" } : (!arc.struct<a : si32, b : si32>) -> si32

    return %elem : si32
    // CHECK-DAG: [[SEVENTEEN:%[^ ]+]] = arc.constant 17 : si32
    // CHECK: return [[SEVENTEEN]] : si32
  }
}

// -----

module @toplevel {
  func @main(%struct : !arc.struct<a : si32, b : si32>) -> si32 {
    %elem = "arc.struct_access"(%struct) { field = "b" } : (!arc.struct<a : si32, b : si32>) -> si32

    return %elem : si32
    // CHECK-DAG: [[STRUCTACCESS:%[^ ]+]] = "arc.struct_access"
    // CHECK: return [[STRUCTACCESS]] : si32
  }
}

