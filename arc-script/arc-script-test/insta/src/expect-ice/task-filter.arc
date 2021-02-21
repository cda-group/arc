task Filter(p: fun(i32) -> bool) ~i32 -> ~i32 {
    on event => {
        if p(event) {
            emit event
        } else {
            ()
        }
    }
}

fun main(stream0: ~i32) -> ~i32 {
    let stream1 = Filter(|x| x % 2 == 0) (stream0) in
    stream1
}
