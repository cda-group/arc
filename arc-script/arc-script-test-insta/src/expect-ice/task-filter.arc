task Filter(p: fun(i32) -> bool) <Input(i32)> -> <Output(i32)> {
    on Input(event) => {
        if p(event) {
            emit Output(event)
        } else {
            ()
        }
    }
}

fun main(stream0: ~i32) -> ~i32 {
    let stream1 = Filter(|x| x % 2 == 0) (stream0) in
    stream1
}
