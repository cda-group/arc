task Identity() ~i32 -> ~i32 {
    on event => emit event
}

fun main(input: ~i32) -> ~i32 {
    let output = Identity() (input) in
    output
}
