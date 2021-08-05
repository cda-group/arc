task Repeat(v: i32, dur: duration): () -> (Output(~i32)) {
    every dur {
        emit Output(v)
    };
}

fun main(): ~i32 {
    val stream' = Repeat(1, 30s) ();
    stream'
}
