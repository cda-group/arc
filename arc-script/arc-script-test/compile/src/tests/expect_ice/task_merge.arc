task Merge(): (A(~i32), B(~i32)) -> (C(~i32)) {
    on {
        A(event) => emit C(event),
        B(event) => emit C(event),
    }
}

fun main(a: ~i32, b: ~i32): ~i32 {
    val c' = Scan() (a, b);
    c'
}
