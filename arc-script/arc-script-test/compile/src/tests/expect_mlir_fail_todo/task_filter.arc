task Filter(p: fun(i32): bool) ~i32 by i32 -> ~i32 by i32 {
    on event by key => {
        if p(event) {
            emit event by key
        } else {
            unit
        }
    }
}

fun main(stream0: ~i32 by i32): ~i32 by i32 {
    let stream1 = Filter(fun(x): x % 2 == 0) (stream0) in
    stream1
}
