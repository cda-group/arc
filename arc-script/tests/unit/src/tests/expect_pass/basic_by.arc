# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir

fun test(): i32 {
    val x = 1 by 2;
    val v = x.value;
    val k = x.key;
    v + k
}
