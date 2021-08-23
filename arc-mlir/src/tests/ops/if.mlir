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
    // expected-note@+1 {{see current operation:}}
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

    "arc.if"(%a) ( {
    // expected-error@+2 {{'arc.block.result' op result type does not match the type of the parent: expected 'f64' but found 'f32'}}
    // expected-note@+1 {{see current operation:}}
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
    // expected-note@+1 {{see current operation:}}
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
    // expected-note@+1 {{see current operation:}}
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
    // expected-note@+1 {{see current operation:}}
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
    // expected-note@+1 {{see current operation:}}
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

    // expected-error@+2 {{'arc.if' op expects terminators to be 'arc.loop.break', 'arc.return' or'arc.block.result' operations}}
    // expected-note@+1 {{see current operation}}
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
    "arc.if"(%a) ( {
      // expected-error@+2 {{'arc.block.result' op result type does not match the type of the parent: expected 'f64' but found 'f32'}}
      // expected-note@+1 {{see current operation:}}
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
    "arc.if"(%a) ( {
       // expected-error@+2 {{'arc.block.result' op result type does not match the type of the parent: expected 'f64' but found 'f32'}}
       // expected-note@+1 {{see current operation:}}
      "arc.block.result"(%b) : (f32) -> ()
    },  {
      "arc.block.result"(%c) : (f32) -> ()
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
    "arc.if"(%a) ( {
      // expected-error@+2 {{'arc.block.result' op cannot return a result from an 'arc.if' without result}}
      // expected-note@+1 {{see current operation: }}
      "arc.block.result"(%b) : (f32) -> ()
    },  {
      "arc.block.result"(%c) : (f32) -> ()
    }) : (i1) -> ()
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f32
    "arc.if"(%a) ( {
       // expected-error@+2 {{'arc.block.result' op cannot return more than one result}}
       // expected-note@+1 {{see current operation}}
      "arc.block.result"(%b, %c) : (f32, f32) -> ()
    },  {
      "arc.block.result"(%c) : (f32) -> ()
    }) : (i1) -> ()
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = constant 3.14 : f32
    %c = constant 0.693 : f32
    // expected-error@+2 {{'arc.if' op cannot return more than one result}}
    // expected-note@+1 {{see current operation}}
    "arc.if"(%a) ( {
      "arc.block.result"(%b,%b) : (f32,f32) -> ()
    },  {
      "arc.block.result"(%c,%b) : (f32,f32) -> ()
    }) : (i1) -> (f32,f32)
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = arc.constant 66 : ui64
    "arc.if"(%a) ( {
      "arc.block.result"(%b) : (ui64) -> ()
    },  {
      // expected-error@+2 {{'arc.block.result' op the number of values returned does not match parent: expected 1 but found 0 values}}
      // expected-note@+1 {{see current operation}}
      "arc.block.result"() : () -> ()
    }) : (i1) -> ui64
    return
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = arc.constant 66 : ui64
    %c = arc.constant 7 : ui64
    "arc.if"(%a) ( {
      "arc.block.result"(%b) : (ui64) -> ()
    },  {
       "arc.return"() : () -> ()
    }) : (i1) -> ui64
    return
  }
}

// -----

module @toplevel {
  func @main() -> ui64 {
    %a = constant 0 : i1
    %b = arc.constant 66 : ui64
    %c = arc.constant 7 : ui64
    "arc.if"(%a) ( {
      "arc.block.result"() : () -> ()
    },  {
       "arc.return"(%c) : (ui64) -> ()
    }) : (i1) -> ()
    return %b : ui64
  }
}

// -----

module @toplevel {
  func @main() -> ui64 {
    %a = constant 0 : i1
    %b = arc.constant 66 : ui64
    %c = arc.constant 7 : si64
    "arc.if"(%a) ( {
      "arc.block.result"() : () -> ()
    },  {
       // expected-error@+2 {{'arc.return' op result type does not match the type of the function: expected 'ui64' but found 'si64'}}
       // expected-note@+1 {{see current operation}}
       "arc.return"(%c) : (si64) -> ()
    }) : (i1) -> ()
    return %b : ui64
  }
}

// -----

module @toplevel {
  func @main() {
    %a = constant 0 : i1
    %b = arc.constant 66 : ui64
    %c = arc.constant 7 : ui64
    "arc.if"(%a) ( {
      "arc.block.result"() : () -> ()
    },  {
       // expected-error@+2 {{'arc.return' op cannot return a value from a void function}}
       // expected-note@+1 {{see current operation}}
       "arc.return"(%c) : (ui64) -> ()
    }) : (i1) -> ()
    return
  }
}

// -----

module @toplevel {
  func @main() -> ui64 {
    %a = constant 0 : i1
    %b = arc.constant 66 : ui64
    "arc.if"(%a) ( {
      "arc.block.result"() : () -> ()
    },  {
       // expected-error@+2 {{'arc.return' op operation must return a 'ui64' value}}
       // expected-note@+1 {{see current operation}}
       "arc.return"() : () -> ()
    }) : (i1) -> ()
    return %b : ui64
  }
}
