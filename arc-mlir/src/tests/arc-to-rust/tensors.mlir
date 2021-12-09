// RUN: arc-mlir-rust-test %t %s -rustinclude %s.rust-tests
// RUN: arc-mlir-rust-test %t-canon %s -rustinclude %s.rust-tests -canonicalize
// RUN: arc-mlir-rust-test %t-roundtrip-scf %s -rustinclude %s.rust-tests -canonicalize -remove-scf -canonicalize -to-scf -canonicalize

module @arctorusttensors {
  func @in_out_0(%0 : tensor<4xsi32>) -> tensor<4xsi32> {
    return %0 : tensor<4xsi32>
  }
  func @in_out_1(%0 : tensor<4x5xsi32>) -> tensor<4x5xsi32> {
    return %0 : tensor<4x5xsi32>
  }
  func @in_out_2(%0 : tensor<?xsi32>) -> tensor<?xsi32> {
    return %0 : tensor<?xsi32>
  }

  func @make_0() -> tensor<1xf32> {
    %a = arith.constant 0.0 : f32

    %0 = "arc.make_tensor"(%a) : (f32)
       	 		       	       	     -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }

  func @make_1() -> tensor<2xf32> {
    %a = arith.constant 0.0 : f32
    %b = arith.constant 1.0 : f32

    %0 = "arc.make_tensor"(%a, %b) : (f32, f32)
       	 		       	       	     -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  func @make_2() -> tensor<3xf32> {
    %a = arith.constant 0.0 : f32
    %b = arith.constant 1.0 : f32
    %c = arith.constant 2.0 : f32

    %0 = "arc.make_tensor"(%a, %b, %c) : (f32, f32, f32)
       	 		       	       	     -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }

  func @make_3() -> tensor<2x3x4xf32> {
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32
    %v3 = arith.constant 3.0 : f32
    %v4 = arith.constant 4.0 : f32
    %v5 = arith.constant 5.0 : f32
    %v6 = arith.constant 6.0 : f32
    %v7 = arith.constant 7.0 : f32
    %v8 = arith.constant 8.0 : f32
    %v9 = arith.constant 9.0 : f32
    %v10 = arith.constant 10.0 : f32
    %v11 = arith.constant 11.0 : f32
    %v12 = arith.constant 12.0 : f32
    %v13 = arith.constant 13.0 : f32
    %v14 = arith.constant 14.0 : f32
    %v15 = arith.constant 15.0 : f32
    %v16 = arith.constant 16.0 : f32
    %v17 = arith.constant 17.0 : f32
    %v18 = arith.constant 18.0 : f32
    %v19 = arith.constant 19.0 : f32
    %v20 = arith.constant 20.0 : f32
    %v21 = arith.constant 21.0 : f32
    %v22 = arith.constant 22.0 : f32
    %v23 = arith.constant 23.0 : f32


    %0 = "arc.make_tensor"(%v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7, %v8, %v9, %v10, %v11, %v12, %v13, %v14, %v15, %v16, %v17, %v18, %v19, %v20, %v21, %v22, %v23 ) : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
       	 		       	       	     -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

}
