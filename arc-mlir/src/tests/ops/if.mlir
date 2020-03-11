// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f64
    %c = constant 0.693 : f64

    "arc.if"(%a) ( {
      "arc.yield"(%b) : (f64) -> f64
    },  {
      "arc.yield"(%c) : (f64) -> f64
    }) : (i1) -> f64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %f = constant 3.14 : f64

    // expected-error@+1 {{'arc.yield' op yield only terminates 'arc.if' or 'arc.for' regions}}
    "arc.yield"(%f) : (f64) -> f64
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f64

    "arc.if"(%a) ( {
    // expected-error@+1 {{'arc.yield' op requires the same type for all operands and results}}
      "arc.yield"(%b) : (f32) -> f64
    },  {
      "arc.yield"(%c) : (f64) -> f64
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
    // expected-error@+1 {{'arc.if' op has incorrect number of regions: expected 2 but found 1}}
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
    // expected-error@+1 {{'arc.if' op has incorrect number of regions: expected 2 but found 3}}
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
    // expected-error@+1 {{'arc.if' op region #0 ('thenRegion') failed to verify constraint: region with 1 blocks}}
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
    // expected-error@+1 {{'arc.if' op region #1 ('elseRegion') failed to verify constraint: region with 1 blocks}}
    "arc.if"(%a) ({
      "arc.yield"(%b) : (f64) -> f64
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

    // expected-error@+2 {{'arc.if' op expects regions to end with 'arc.yield', found 'arc.make_tuple'}}
    // expected-note@+1 {{in custom textual format, the absence of terminator implies 'arc.yield'}}
    "arc.if"(%a) ( {
      "arc.yield"(%b) : (f64) -> f64
      %1 = "arc.make_tuple"(%c, %c) : (f64, f64) -> tuple<f64,f64>
    },  {
      "arc.yield"(%c) : (f64) -> f64
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

    // expected-error@+1 {{'arc.if' op result type does not match the type of the parent: expected 'f64' but found 'f32'}}
    "arc.if"(%a) ( {
      "arc.yield"(%b) : (f32) -> f32
    },  {
      "arc.yield"(%c) : (f64) -> f64
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
    // expected-error@+1 {{'arc.if' op result type does not match the type of the parent: expected 'f64' but found 'f32'}}
    "arc.if"(%a) ( {
      "arc.yield"(%b) : (f32) -> f32
    },  {
      "arc.yield"(%c) : (f32) -> f32
    }) : (i1) -> f64
    return
  }
}
