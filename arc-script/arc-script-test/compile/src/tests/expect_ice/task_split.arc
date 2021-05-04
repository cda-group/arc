task Split(p: fun(i32): bool): ~i32 -> (A(~i32), B(~i32)) {
    on event => {
        if p(event) {
            emit A(event)
        } else {
            emit B(event)
        }
    }
}

fun main(a: ~i32): (~i32, ~i32) {
    let (b, c) = Split(fun(x): x % 2) (a) in
    (b, c)
}
