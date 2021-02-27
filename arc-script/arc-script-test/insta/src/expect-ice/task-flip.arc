task Flip() (A0(~i32), B0(~i32)) -> (A1(~i32), B1(~i32)) {
    on {
        A0(event) => emit B1(event),
        B0(event) => emit A1(event),
    }
}

fun main(a: ~i32, b: ~i32) -> (~i32, ~i32) {
    Flip() (a, b)
}
