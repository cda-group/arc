# RUN: arc-script run --output=MLIR %s | arc-mlir -rustcratename expectpassif -arc-to-rust

fun test(): i32 {
    val c0 = true;
    val c1 = false;
    val v = 3;
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
