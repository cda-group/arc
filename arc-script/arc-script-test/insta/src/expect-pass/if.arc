# RUN: arc-script run --output=MLIR %s | arc-mlir -rustcratename expectpassif -arc-to-rust -crate %t && arc-cargo test -j 1 --manifest-path=%t/expectpassif/Cargo.toml

fun test() -> i32 {
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
