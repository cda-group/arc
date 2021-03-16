task Map(f: fun(i32) -> i32) ~i32 -> ~i32 {
    on event => emit f(event)
}

fun main(input: ~i32) -> ~i32 {
    let output = Map(|x| x + 1) (input) in
    output
}