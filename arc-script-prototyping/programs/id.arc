def id[A](x: A): A {
    x
}

def foo() {
    val x = id(1);
}
