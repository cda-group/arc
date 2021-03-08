extern fun increment(x: i32) -> i32

task Adder() ~i32 -> ~i32 {
    extern fun addition(x: i32, y: i32) -> i32

    on event => emit addition(event, event)
}

fun pipe(s: ~i32) -> ~i32 {
    if increment(1) == 2 {
        Adder() (s)
    } else {
        Adder() (s)
    }
}
