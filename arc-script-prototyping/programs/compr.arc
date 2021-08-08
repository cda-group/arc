fun plus_one(stream: ~i32): ~i32 {
    [emit event+1; for event in stream; if event != 0]
}
