task Identity() (Input(~i32)) -> (Output(~i32)) {
    on Input(event) => emit Output(event)
}

fun main(input: ~i32): ~i32 {
    let output = Identity() (input) in
    output
}
