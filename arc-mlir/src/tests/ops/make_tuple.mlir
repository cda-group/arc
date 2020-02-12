// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1
    %c = constant 0 : i1

    %1 = "arc.make_tuple"(%a, %b, %c) : (i1, i1, i1) -> tuple<i1,i1,i1>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1
    // expected-error@+1 {{'arc.make_tuple' op operand types do not match, found 'i1' but expected 'f32'}}
    %1 = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,f32>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant -3.40282347E+38 : f32 // expected-note {{prior use here}}
    %b = constant 0 : i1
    // expected-error@+1 {{use of value '%a' expects different type than prior uses: 'i1' vs 'f32'}}
    %0 = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1,i1>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 1 : i1
    // expected-error@+1 {{'arc.make_tuple' op result does not match the number of operands: found 2 but expected 1 operands}}
    %1 = "arc.make_tuple"(%a, %b) : (i1, i1) -> tuple<i1>
    return
  }
}

// -----

module @toplevel {
  func @main() {
    // expected-error@+1 {{'arc.make_tuple' op tuple must contain at least one element}}
    %1 = "arc.make_tuple"() : () -> tuple<>
    return
  }
}
