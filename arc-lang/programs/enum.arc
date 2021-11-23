
enum Foo[T] {
    Bar(T)
}

def qux() {
    Foo::Bar(1);
    Foo::Bar(1.0);
}
