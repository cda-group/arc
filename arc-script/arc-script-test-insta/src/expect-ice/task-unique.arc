task Unique() (Input(i32)) -> (Output(i32)) {
    state map: {i32} = {};
    on Input(event) => {
        if !contains(map, event) {
            insert(map, event)
            emit Output(map)
        }
    }
}

fun main(stream: ~i32) -> ~i32 {
    let stream' = Unique() (stream) in
    stream'
}
