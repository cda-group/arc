// RUN: arc-mlir %s -split-input-file -verify-diagnostics

// -----

module @toplevel {
  func @main(%arg0: !arc.unknown) {
    return
  }
}

