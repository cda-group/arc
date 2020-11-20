// RUN: arc-mlir --canonicalize %s | FileCheck %s

//CHECK-DAG: [[C200:%[^ ]+]] = constant 200 : i32
//CHECK-DAG: [[C300:%[^ ]+]] = constant 300 : i32
//CHECK-DAG: [[C400:%[^ ]+]] = constant 400 : i32
//CHECK-DAG: [[C500:%[^ ]+]] = constant 500 : i32
//CHECK-DAG: [[C900:%[^ ]+]] = constant 900 : i32
//CHECK-DAG: [[C1000:%[^ ]+]] = constant 1000 : i32

module @toplevel {
  func @main(%arg0: i1) -> () {
    %false = constant 0 : i1
    %true = constant 1 : i1

    %r0 = "arc.if"(%false) ( {
      %b = constant 100 : i32
      "arc.block.result"(%b) : (i32) -> ()
    },  {
      %c = constant 200 : i32
      "arc.block.result"(%c) : (i32) -> ()
    }) : (i1) -> i32
    "arc.keep"(%r0) : (i32) -> ()
//CHECK: "arc.keep"([[C200]])

    %r1 = "arc.if"(%true) ( {
      %d = constant 300 : i32
      "arc.block.result"(%d) : (i32) -> ()
    },  {
      %e = constant 400 : i32
      "arc.block.result"(%e) : (i32) -> ()
    }) : (i1) -> i32
    "arc.keep"(%r1) : (i32) -> ()
//CHECK: "arc.keep"([[C300]])

    %r2 = "arc.if"(%arg0) ( {
      %d = constant 300 : i32
      "arc.block.result"(%d) : (i32) -> ()
    },  {
      %e = constant 400 : i32
      "arc.block.result"(%e) : (i32) -> ()
    }) : (i1) -> i32
    "arc.keep"(%r2) : (i32) -> ()
//CHECK: "arc.block.result"([[C300]])
//CHECK: "arc.block.result"([[C400]])

    %r3 = "arc.if"(%true) ( {
      %d = constant 500 : i32
      "arc.block.result"(%d) : (i32) -> ()
    },  {
      %e = "arc.if"(%arg0) ( {
        %d = constant 600 : i32
        "arc.block.result"(%d) : (i32) -> ()
      },  {
        %e = constant 700 : i32
        "arc.block.result"(%e) : (i32) -> ()
      }) : (i1) -> i32
      "arc.block.result"(%e) : (i32) -> ()
    }) : (i1) -> i32
    "arc.keep"(%r3) : (i32) -> ()
//CHECK: "arc.keep"([[C500]])

    %r4 = "arc.if"(%false) ( {
      %d = constant 800 : i32
      "arc.block.result"(%d) : (i32) -> ()
    },  {
      %e = "arc.if"(%arg0) ( {
        %d = constant 900 : i32
        "arc.block.result"(%d) : (i32) -> ()
      },  {
        %e = constant 1000 : i32
        "arc.block.result"(%e) : (i32) -> ()
      }) : (i1) -> i32
      "arc.block.result"(%e) : (i32) -> ()
    }) : (i1) -> i32
    "arc.keep"(%r4) : (i32) -> ()
//CHECK: "arc.block.result"([[C900]])
//CHECK: "arc.block.result"([[C1000]])
    return
  }
}
