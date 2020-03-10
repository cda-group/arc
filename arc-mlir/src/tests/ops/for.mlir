// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main() {

    %b0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %t0 = constant dense<[1,2,3]> : tensor<3xi32>

    %b4 = "arc.for"(%b0, %t0) ({
      ^iter(%b1: !arc.appender<i32>, %i1: index, %e1: i32):

      %e2 = addi %e1, %e1 : i32
      %b2 = "arc.merge"(%b1, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
      %b3 = "arc.merge"(%b2, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>

      "arc.yield"(%b3) : (!arc.appender<i32>) -> !arc.appender<i32>
    }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

    %t1 = "arc.result"(%b4) : (!arc.appender<i32>) -> tensor<i32>

    return
  }
}

// -----

module @toplevel {
  func @main() {

    %b0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %t0 = constant dense<[1,2,3]> : tensor<3xi32>

    %b5 = "arc.for"(%b0, %t0) ({
      ^iter(%b1: !arc.appender<i32>, %i1: index, %e1: i32):

      %b4 = "arc.for"(%b1, %t0) ({
        ^iter(%b2: !arc.appender<i32>, %i2: index, %e2: i32):

        %e3 = addi %e2, %e2 : i32
        %b3 = "arc.merge"(%b2, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
        "arc.yield"(%b3) : (!arc.appender<i32>) -> !arc.appender<i32>
      }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

      "arc.yield"(%b4) : (!arc.appender<i32>) -> !arc.appender<i32>
    }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

    %t1 = "arc.result"(%b5) : (!arc.appender<i32>) -> tensor<i32>

    return
  }
}

// -----

module @toplevel {
  func @main() {

    %b0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %t0 = constant dense<[1,2,3]> : tensor<3xi32>


    // expected-error@+1 {{'arc.for' op block argument #1 is not an index type, found: 'i32' but expected 'index'}}
    %b4 = "arc.for"(%b0, %t0) ({
      ^iter(%b1: !arc.appender<i32>, %i1: i32, %e1: i32):

      %e2 = addi %e1, %e1 : i32
      %b2 = "arc.merge"(%b1, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
      %b3 = "arc.merge"(%b2, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>

      "arc.yield"(%b3) : (!arc.appender<i32>) -> !arc.appender<i32>
    }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

    %t1 = "arc.result"(%b4) : (!arc.appender<i32>) -> tensor<i32>

    return
  }
}

// -----

module @toplevel {
  func @main() {

    %b0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %t0 = constant dense<[1,2,3]> : tensor<3xi32>


    // expected-error@+1 {{'arc.for' op block takes incorrect number of arguments, found: 2 but expected 3}}
    %b4 = "arc.for"(%b0, %t0) ({
      ^iter(%b1: !arc.appender<i32>, %e1: i32):

      %e2 = addi %e1, %e1 : i32
      %b2 = "arc.merge"(%b1, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
      %b3 = "arc.merge"(%b2, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>

      "arc.yield"(%b3) : (!arc.appender<i32>) -> !arc.appender<i32>
    }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

    %t1 = "arc.result"(%b4) : (!arc.appender<i32>) -> tensor<i32>

    return
  }
}

// -----

module @toplevel {
  func @main() {

    %b0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %t0 = constant dense<[1,2,3]> : tensor<3xi32>

    // expected-error@+1 {{'arc.for' op failed to verify that result must have exactly one use}}
    %b4 = "arc.for"(%b0, %t0) ({
      ^iter(%b1: !arc.appender<i32>, %i1: index, %e1: i32):

      %e2 = addi %e1, %e1 : i32
      %b2 = "arc.merge"(%b1, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
      %b3 = "arc.merge"(%b2, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>

      "arc.yield"(%b3) : (!arc.appender<i32>) -> !arc.appender<i32>
    }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

    return
  }
}

// -----

module @toplevel {
  func @main() {

    %b0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %t0 = constant dense<[1,2,3]> : tensor<3xi32>

    // expected-error@+2 {{'arc.for' op expects regions to end with 'arc.yield', found 'arc.merge'}}
    // expected-note@+1 {{in custom textual format, the absence of terminator implies 'arc.yield'}}
    %b4 = "arc.for"(%b0, %t0) ({
      ^iter(%b1: !arc.appender<i32>, %i1: index, %e1: i32):

      %e2 = addi %e1, %e1 : i32
      %b2 = "arc.merge"(%b1, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
      %b3 = "arc.merge"(%b2, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>

    }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

    %t1 = "arc.result"(%b4) : (!arc.appender<i32>) -> tensor<i32>

    return
  }
}

// -----

module @toplevel {
  func @main() {

    %b0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %t0 = constant dense<[1,2,3]> : tensor<3xi32>

    // expected-error@+1 {{'arc.for' op block argument #0 must have exactly one use}}
    %b4 = "arc.for"(%b0, %t0) ({
      ^iter(%b1: !arc.appender<i32>, %i1: index, %e1: i32):

      %b3 = "arc.make_appender"() : () -> !arc.appender<i32>

      "arc.yield"(%b3) : (!arc.appender<i32>) -> !arc.appender<i32>
    }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

    %t1 = "arc.result"(%b4) : (!arc.appender<i32>) -> tensor<i32>

    return
  }
}

// -----

module @toplevel {
  func @main() {

    %b0 = "arc.make_appender"() : () -> !arc.appender<i32>
    %t0 = constant dense<[1,2,3]> : tensor<3xi32>

    %b5 = "arc.for"(%b0, %t0) ({
      ^iter(%b1: !arc.appender<i32>, %i1: index, %e1: i32):

      // expected-error@+1 {{'arc.for' op block takes incorrect number of arguments, found: 2 but expected 3}}
      %b4 = "arc.for"(%b1, %t0) ({
        ^iter(%b2: !arc.appender<i32>, %e2: i32):

        %e3 = addi %e2, %e2 : i32
        %b3 = "arc.merge"(%b2, %e2) : (!arc.appender<i32>, i32) -> !arc.appender<i32>
        "arc.yield"(%b3) : (!arc.appender<i32>) -> !arc.appender<i32>
      }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

      "arc.yield"(%b4) : (!arc.appender<i32>) -> !arc.appender<i32>
    }) : (!arc.appender<i32>, tensor<3xi32>) -> !arc.appender<i32>

    %t1 = "arc.result"(%b5) : (!arc.appender<i32>) -> tensor<i32>

    return
  }
}
