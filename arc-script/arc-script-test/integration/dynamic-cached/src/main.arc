fun pipe(s: ~i32) -> ~i32 {
    Identity() (s)
}

task Identity() ~i32 -> ~i32 {
    extern fun rust_method(x: i32) -> i32
    fun arc_method(x: i32) -> i32 {
        x - 5
    }
    on event => emit rust_method(event)
}
