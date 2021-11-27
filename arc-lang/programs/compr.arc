def plus_one(stream: Stream[i32]): Stream[i32] {
    [emit event+1; for event in stream; if event != 0]
}
