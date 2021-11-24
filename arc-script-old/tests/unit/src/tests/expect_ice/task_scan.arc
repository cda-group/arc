task Scan(init: i32, f: fun(i32, i32): i32) ~i32 -> ~i32 {
    state agg: i32 = init;
    on event => {
        agg = f(agg, event);
        emit agg
    }
}

fun main(stream: ~i32): ~i32 {
    val stream' = Scan(0) (stream);
    stream'
}
