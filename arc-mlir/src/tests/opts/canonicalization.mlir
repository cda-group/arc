// RUN: arc-mlir --canonicalize %s | FileCheck %s
module @toplevel {
  func @main(%arg0 : i64) {
    %cst_0 = constant -3.40282347E+38 : f32
// CHECK: {{%[^ ]+}} = constant dense<-3.40282347E+38> : tensor<1xf32>
    %0 = "arc.make_vector"(%cst_0) : (f32) -> tensor<1xf32>
    "arc.keep"(%0) : (tensor<1xf32>) -> ()

    %cst_1 = constant 3.14 : f32
    %cst_2 = constant 0.693 : f32
    %1 = "arc.make_vector"(%cst_0, %cst_1, %cst_2) : (f32,f32,f32) -> tensor<3xf32>
// CHECK: {{%[^ ]+}} = constant dense<[-3.40282347E+38, 3.140000e+00, 6.930000e-01]> : tensor<3xf32>
    "arc.keep"(%1) : (tensor<3xf32>) -> ()

    %cst_3 = constant 3 : i64
    %cst_4 = constant 4 : i64
    %cst_5 = constant 5 : i64
    %2 = "arc.make_vector"(%cst_3, %cst_4, %cst_5) : (i64,i64,i64) -> tensor<3xi64>
// CHECK: {{%[^ ]+}} = constant dense<[3, 4, 5]> : tensor<3xi64>
    "arc.keep"(%2) : (tensor<3xi64>) -> ()

    %3 = "arc.make_vector"(%arg0, %cst_4, %cst_5) : (i64,i64,i64) -> tensor<3xi64>
    "arc.keep"(%3) : (tensor<3xi64>) -> ()
// Check that canonicalization did not fold the non-constant argument
// CHECK: {{%[^ ]+}} = "arc.make_vector"(%arg0,
    return
  }
}
