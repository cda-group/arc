// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f64
    %c = constant 0.693 : f64

    "arc.if"(%a) ( {
      "arc.block.result"(%b) : (f64) -> ()
    },  {
      "arc.block.result"(%c) : (f64) -> ()
    }) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %f = constant 3.14 : f64

    // expected-error@+2 {{'arc.block.result' op expects parent op 'arc.if'}}
    // expected-note@+1 {{see current operation: "arc.block.result"}}
    "arc.block.result"(%f) : (f64) -> ()
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f64

    // expected-error@+2 {{'arc.if' op result type does not match the type of the parent: expected 'f64' but found 'f32'}}
    // expected-note@+1 {{see current operation: %0 = "arc.if"(%false)}}
    "arc.if"(%a) ( {
    "arc.block.result"(%b) : (f32) -> ()
    },  {
      "arc.block.result"(%c) : (f64) -> ()
    }) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f64
    // expected-error@+1 {{expected '{' to begin a region}}
    "arc.if"(%a) () : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f64
    // expected-error@+2 {{'arc.if' op expected 2 regions}}
    // expected-note@+1 {{see current operation: %0 = "arc.if"}}
    "arc.if"(%a) ({}) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f64
    // expected-error@+2 {{'arc.if' op expected 2 regions}}
    // expected-note@+1 {{see current operation: %0 = "arc.if"}}
    "arc.if"(%a) ({},{},{}) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f64
    // expected-error@+2 {{'arc.if' op region #0 ('thenRegion') failed to verify constraint: region with 1 blocks}}
    // expected-note@+1 {{see current operation: %0 = "arc.if"}}
    "arc.if"(%a) ({},{}) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f64
    %c = constant 0.693 : f64
    // expected-error@+2 {{'arc.if' op region #1 ('elseRegion') failed to verify constraint: region with 1 blocks}}
    // expected-note@+1 {{see current operation: %0 = "arc.if"}}
    "arc.if"(%a) ({
      "arc.block.result"(%b) : (f64) -> ()
    },{}) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f64
    %c = constant 0.693 : f64

    // expected-error@+3 {{'arc.if' op expects regions to end with 'arc.block.result', found 'arc.make_tuple'}}
    // expected-note@+2 {{in custom textual format, the absence of terminator implies 'arc.block.result'}}
    // expected-note@+1 {{see current operation: %0 = "arc.if"}}
    "arc.if"(%a) ( {
      "arc.block.result"(%b) : (f64) -> ()
      %1 = "arc.make_tuple"(%c, %c) : (f64, f64) -> tuple<f64,f64>
    },  {
      "arc.block.result"(%c) : (f64) -> ()
    }) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f64

    // expected-error@+2 {{'arc.if' op result type does not match the type of the parent: expected 'f64' but found 'f32'}}
    // expected-note@+1 {{see current operation: %0 = "arc.if"}}
    "arc.if"(%a) ( {
      "arc.block.result"(%b) : (f32) -> ()
    },  {
      "arc.block.result"(%c) : (f64) -> ()
    }) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f32
    // expected-error@+2 {{'arc.if' op result type does not match the type of the parent: expected 'f64' but found 'f32'}}
    // expected-note@+1 {{see current operation: %0 = "arc.if"}}
    "arc.if"(%a) ( {
      "arc.block.result"(%b) : (f32) -> ()
    },  {
      "arc.block.result"(%c) : (f32) -> ()
    }) : (i1) -> f64
    return
  }
}
