# RUN: arc-script run --output=MLIR %s | arc-mlir -rustcratename expectpassif -arc-to-rust

fun test(): i32 {
    let c0 = true in
    let c1 = false in
    let v = 3 in
    if c0 { 
      if c1 {
          v
      } else {
          v
      }
    } else {
      if c1 {
          v
      } else {
          v
      }
    }
}
