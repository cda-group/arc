task Tee() (A(i32)) -> (B(i32), C(i32)) {
    on A(event) => {
        emit C(event);
        emit C(event)
    }
}

fun main(a: ~i32) -> (~i32, ~i32) {
    let (b, c) = Tee() (a) in
    (b, c)
}

