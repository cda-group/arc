// RUN: arc-mlir %s -split-input-file -verify-diagnostics

module @toplevel {
  func.func @make_0() -> tensor<?x2xf32> {
    %a = arith.constant 0.0 : f32
    %b = arith.constant 1.0 : f32
    %c = arith.constant 2.0 : f32
    %d = arith.constant 3.0 : f32

    // expected-error@+2 {{'arc.make_tensor' op result #0 must be statically shaped tensor of arc-tensor-element values, but got 'tensor<?x2xf32>'}}
    // expected-note@+1 {{see current operation}}
    %0 = "arc.make_tensor"(%a, %b, %c, %d) : (f32, f32, f32, f32)
       	 		       	       	     -> tensor<?x2xf32>
    return %0 : tensor<?x2xf32>
  }
}

// -----

module @toplevel {
  func.func @make_0() -> tensor<*xf32> {
    %a = arith.constant 0.0 : f32
    %b = arith.constant 1.0 : f32
    %c = arith.constant 2.0 : f32
    %d = arith.constant 3.0 : f32

    // expected-error@+2 {{'arc.make_tensor' op result #0 must be statically shaped tensor}}
    // expected-note@+1 {{see current operation:}}
    %0 = "arc.make_tensor"(%a, %b, %c, %d) : (f32, f32, f32, f32)
       	 		       	       	     -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }
}

// -----

func.func @make_0() -> tensor<2x2xf32> {
    %a = arith.constant 0.0 : f32
    %b = arith.constant 1.0 : f32
    %c = arith.constant 2.0 : f64
    %d = arith.constant 3.0 : f32

    // expected-error@+2 {{'arc.make_tensor' op requires the same element type for all operands and results}}
    // expected-note@+1 {{see current operation:}}
    %0 = "arc.make_tensor"(%a, %b, %c, %d) : (f32, f32, f64, f32)
       	 		       	       	     -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

// -----

func.func @make_0() -> tensor<1x2xf32> {
    %a = arith.constant 0.0 : f32
    %b = arith.constant 1.0 : f32
    %c = arith.constant 2.0 : f32
    %d = arith.constant 3.0 : f32

    // expected-error@+2 {{'arc.make_tensor' op wrong number of operands: expected 2 but found 4 operands}}
    // expected-note@+1 {{see current operation:}}
    %0 = "arc.make_tensor"(%a, %b, %c, %d) : (f32, f32, f32, f32)
       	 		       	       	     -> tensor<1x2xf32>
    return %0 : tensor<1x2xf32>
  }

// -----
