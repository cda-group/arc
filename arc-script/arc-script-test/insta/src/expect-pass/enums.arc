# XFAIL: *
# RUN: arc-script run --output=MLIR %s | arc-mlir

enum Foo {
  Bar(u32),
  Baz(f32),
}

fun main() {
    let a = 200u32 in
    let b = 2.0f32 in

    let c = Foo::Bar(a) in
    let d = unwrap[Foo::Bar](c) in

    let e = enwrap[Foo::Baz](b) in
    let f = is[Foo::Baz](e) in

    unit
}
