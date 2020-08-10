// RUN: arc-mlir -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml
// RUN: arc-mlir -canonicalize -arc-to-rust -crate %t -extra-rust-trailer %s.rust-tests %s && cargo test -j 1 --manifest-path=%t/toplevel/Cargo.toml

module @toplevel {
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
    %a = constant 0.0 : f32

    %0 = "arc.make_tensor"(%a) : (f32)
       	 		       	       	     -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }

  func @make_1() -> tensor<2xf32> {
    %a = constant 0.0 : f32
    %b = constant 1.0 : f32

    %0 = "arc.make_tensor"(%a, %b) : (f32, f32)
       	 		       	       	     -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  func @make_2() -> tensor<3xf32> {
    %a = constant 0.0 : f32
    %b = constant 1.0 : f32
    %c = constant 2.0 : f32

    %0 = "arc.make_tensor"(%a, %b, %c) : (f32, f32, f32)
       	 		       	       	     -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }

  func @make_3() -> tensor<2x3x4xf32> {
    %v0 = constant 0.0 : f32
    %v1 = constant 1.0 : f32
    %v2 = constant 2.0 : f32
    %v3 = constant 3.0 : f32
    %v4 = constant 4.0 : f32
    %v5 = constant 5.0 : f32
    %v6 = constant 6.0 : f32
    %v7 = constant 7.0 : f32
    %v8 = constant 8.0 : f32
    %v9 = constant 9.0 : f32
    %v10 = constant 10.0 : f32
    %v11 = constant 11.0 : f32
    %v12 = constant 12.0 : f32
    %v13 = constant 13.0 : f32
    %v14 = constant 14.0 : f32
    %v15 = constant 15.0 : f32
    %v16 = constant 16.0 : f32
    %v17 = constant 17.0 : f32
    %v18 = constant 18.0 : f32
    %v19 = constant 19.0 : f32
    %v20 = constant 20.0 : f32
    %v21 = constant 21.0 : f32
    %v22 = constant 22.0 : f32
    %v23 = constant 23.0 : f32


    %0 = "arc.make_tensor"(%v0, %v1, %v2, %v3, %v4, %v5, %v6, %v7, %v8, %v9, %v10, %v11, %v12, %v13, %v14, %v15, %v16, %v17, %v18, %v19, %v20, %v21, %v22, %v23 ) : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
       	 		       	       	     -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

}
