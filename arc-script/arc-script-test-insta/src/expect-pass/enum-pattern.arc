enum Option {
    Some(i32),
    None,
}

fun main() {
    if let Option::Some(x) = Option::Some(5) {
        ()
    } else {
        ()
    }
}