task Split(p: fun(i32) -> bool) (A(i32)) -> (B(i32), C(i32)) {
    on Input(event) => {
        if p(event) {
            emit B(event)
        } else {
            emit C(event)
        }
    }
}

fun main(a: ~i32) -> (~i32, ~i32) {
    let (b, c) = Split(|x| x % 2) (a) in
    (b, c)
}
