# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir

fun foo(a:{x:i32, y:i32}): i32 {
    a.x + a.y
}

fun bar(a:{y:i32, x:i32}): i32 {
    a.x + a.y
}
