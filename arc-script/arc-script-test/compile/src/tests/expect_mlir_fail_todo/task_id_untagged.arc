task Identity(): ~i32 by i32 -> ~i32 by i32 {
    on event => emit event
}

fun main(input: ~i32 by i32): ~i32 by i32 {
    let output = Identity() (input) in
    output
}
