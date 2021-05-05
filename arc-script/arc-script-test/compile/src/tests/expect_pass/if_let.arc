# RUN: arc-script run --output=MLIR %s | arc-mlir -rustcratename expectpassifval -arc-to-rust
fun test(): i32 {
    val x = 3;
    val y = 5;
    if val ((1, z), 1) = ((3, x), 1) {
        z+y
    } else {
        y+2
    }
}
