enum Option {
    Some(i32),
    None
}

fun main() {
    if let Option::Some(y) = Option::Some(3) {
        ()
    } else {
        ()
    }
}
