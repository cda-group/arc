// RUN: arc-mlir -split-input-file --canonicalize %s | FileCheck %s

module @toplevel {
  func.func @main() -> i1 {
    %a = arith.constant 0 : i1
    %b = arith.constant 1 : i1

    %enum = arc.make_enum(%a : i1) as "a" : !arc.enum<a : i1, b : i1>
    %elem = arc.enum_access "a" in (%enum : !arc.enum<a : i1, b : i1>) : i1

    return %elem : i1
    // CHECK-DAG: [[FALSE:%[^ ]+]] = arith.constant false
    // CHECK: return [[FALSE]] : i1
  }
}

// -----

module @toplevel {
  func.func @main() -> i1 {
    %a = arith.constant 0 : i1
    %b = arith.constant 1 : i1

    %enum = arc.make_enum(%a : i1) as "a" : !arc.enum<a : i1, b : i1>
    %elem = arc.enum_access "b" in (%enum : !arc.enum<a : i1, b : i1>) : i1

    return %elem : i1
    // CHECK-DAG: [[ACCESS:%[^ ]+]] = arc.enum_access
    // CHECK: return [[ACCESS]] : i1
  }
}

// -----

module @toplevel {
  func.func @main() -> si32 {
    %a = arc.constant 17 : si32
    %b = arc.constant 7 : si32

    %enum = arc.make_enum (%a : si32) as "a" : !arc.enum<a : si32, b : si32>

    %elem = arc.enum_access "a" in (%enum : !arc.enum<a : si32, b : si32>) : si32
    return %elem : si32
    // CHECK-DAG: [[SEVENTEEN:%[^ ]+]] = arc.constant 17 : si32
    // CHECK: return [[SEVENTEEN]] : si32
  }
}

// -----

module @toplevel {
  func.func @main() -> si32 {
    %a = arc.constant 17 : si32
    %b = arc.constant 7 : si32

    %enum = arc.make_enum (%b : si32) as "b" : !arc.enum<a : si32, b : si32>

    %elem = arc.enum_access "b" in (%enum : !arc.enum<a : si32, b : si32>) : si32
    return %elem : si32
    // CHECK-DAG: [[SEVEN:%[^ ]+]] = arc.constant 7 : si32
    // CHECK: return [[SEVEN]] : si32
  }
}

// -----

module @toplevel {
  func.func @main(%enum : !arc.enum<a : si32, b : si32>) -> si32 {
    %elem = arc.enum_access "a" in (%enum : !arc.enum<a : si32, b : si32>) : si32

    return %elem : si32
    // CHECK-DAG: [[ENUMACCESS:%[^ ]+]] = arc.enum_access
    // CHECK: return [[ENUMACCESS]] : si32
  }
}

// -----

module @toplevel {
  func.func @main(%enum : !arc.enum<a : si32, b : si32>) -> si32 {
    %elem = arc.enum_access "b" in (%enum : !arc.enum<a : si32, b : si32>) : si32

    return %elem : si32
    // CHECK-DAG: [[ENUMACCESS:%[^ ]+]] = arc.enum_access
    // CHECK: return [[ENUMACCESS]] : si32
  }
}

// -----

module @toplevel {
  func.func @main(%enum : !arc.enum<a : si32, b : si32>) -> i1 {
    %bool = arc.enum_check (%enum : !arc.enum<a : si32, b : si32>) is "a" : i1

    return %bool : i1
    // CHECK-DAG: [[ENUMCHECK:%[^ ]+]] = arc.enum_check
    // CHECK: return [[ENUMCHECK]] : i1
  }
}

// -----

module @toplevel {
  func.func @main() -> i1 {
    %a = arc.constant 17 : si32
    %b = arc.constant 7 : si32

    %enum = arc.make_enum (%b : si32) as "b" : !arc.enum<a : si32, b : si32>
    %bool = arc.enum_check (%enum : !arc.enum<a : si32, b : si32>) is "a" : i1

    return %bool : i1
    // CHECK-DAG: [[FALSE:%[^ ]+]] = arith.constant false
    // CHECK: return [[FALSE]] : i1
  }
}

// -----

module @toplevel {
  func.func @main() -> i1 {
    %a = arc.constant 17 : si32
    %b = arc.constant 7 : si32

    %enum = arc.make_enum (%b : si32) as "b" : !arc.enum<a : si32, b : si32>
    %bool = arc.enum_check (%enum : !arc.enum<a : si32, b : si32>) is "b" : i1

    return %bool : i1
    // CHECK-DAG: [[TRUE:%[^ ]+]] = arith.constant true
    // CHECK: return [[TRUE]] : i1
  }
}
