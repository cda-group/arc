# RUN: arc-script run --output=MLIR %s | arc-mlir

enum Foo {
  Bar(u32),
  Baz(f32),
}

fun main() {

    val a = 200u32;
    val b = 2.0f32;

    val c = Foo::Bar(a);
    val d = unwrap[Foo::Bar](c);

    val e = enwrap[Foo::Baz](b);
    val f = is[Foo::Baz](e);

}
