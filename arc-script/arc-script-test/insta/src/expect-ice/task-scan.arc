task Scan(init: i32, f: fun(i32, i32) -> i32) (Input(i32)) -> (Output(i32)) {
    state agg: i32 = init;
    on Input(event) => {
        let agg' = f(agg, event) in
        agg := agg';
        emit Output(agg')
    }
}

fun main(stream: ~i32) -> ~i32 {
    let stream' = Scan(0) (stream) in
    stream'
}
