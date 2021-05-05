task Filter(p: fun(i32): bool): ~i32 by i32 -> ~i32 by i32 {
    on event by key => {
        if p(event) {
            emit event by key
        }
    };
}

fun main(stream0: ~i32 by i32): ~i32 by i32 {
    stream0 | Filter(fun(x): x % 2 == 0)
}
