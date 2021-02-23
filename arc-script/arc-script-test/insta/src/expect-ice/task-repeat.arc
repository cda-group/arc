task Repeat(v: i32, d: duration) () -> (Output(~i32)) {

    trigger Trigger(unit);

    start_timer();

    fun start_timer() {
        after d |_| emit Trigger(unit)
    }

    on Trigger(_) => {
        start_timer();
        emit Output(v)
    }
}

fun main() -> ~i32 {
    let stream' = Repeat(1, 30s) () in
    stream'
}
