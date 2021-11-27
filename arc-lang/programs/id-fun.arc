def id[A](x: A): A {
    x
}

def foo() {
    val x = id(1);
    val y = id(1.0);
}
