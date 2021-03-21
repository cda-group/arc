task Map(f: fun(i32) -> i32) ~i32 by i32 -> ~i32 by i32 {
    on event by key => emit f(event) by key
}

fun main(input: ~i32 by i32) -> ~i32 by i32 {
    let output = Map(|x| x + 1) (input) in
    output
}
