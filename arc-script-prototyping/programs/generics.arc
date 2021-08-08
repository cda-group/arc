enum List[T] {
    Cons(T, List[T]),
    Nil
}


enum Option[T] {
    Some(T),
    None
}

fun test() {
    val x = Option::Some(5);
    val y = Option::Some(5.0);
    if val Option::Some(z) = x {

    }
}

