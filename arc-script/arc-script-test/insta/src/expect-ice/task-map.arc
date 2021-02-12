task Map(f: fun(i32) -> i32) <Input(i32)> -> <Output(i32)> {
    on Input(event) => emit Output(f(event))
}

fun main(input: ~i32) -> ~i32 {
    let output = Map(|x| x + 1) (input) in
    output
}
