# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir

fun test(): i32 {

    val a = 3;
    val b = fun(x): x + 1;
    val c = fun(x): x - 1;

    val d: i32 = a | b | c;
    # val d = c(b(a));

    d
}
