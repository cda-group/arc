task Tee(): ~i32 -> (A(~i32), B(~i32)) {
    on event => {
        emit A(event);
        emit B(event)
    }
}

fun main(a: ~i32): (~i32, ~i32) {
    val (b, c) = Tee() (a);
    (b, c)
}

