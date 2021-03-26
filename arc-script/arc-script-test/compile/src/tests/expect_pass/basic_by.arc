# RUN: arc-script run --output=MLIR %s | arc-mlir

fun test(): i32 {
    let x = 1 by 2 in
    let v = x.val in
    let k = x.key in
    v + k
}
