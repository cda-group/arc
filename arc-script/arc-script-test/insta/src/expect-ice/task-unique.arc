task Unique() ~i32 -> ~i32 {
    state set: {i32} = {};
    on event => {
        if not contains(map, event) {
            insert(map, event)
            emit event
        }
    }
}

fun main(stream: ~i32) -> ~i32 {
    let stream' = Unique() (stream) in
    stream'
}
