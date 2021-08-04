# RUN: arc-script --no-prelude run --output=MLIR %s | arc-mlir

enum Foo {
    Bar(unit)
}

fun test() {
    val x = Foo::Bar(unit);
    if val Foo::Bar(y) = x {
        y
    } else {
        unit
    }
}
