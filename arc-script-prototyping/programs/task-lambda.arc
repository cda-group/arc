def test(s: ~i32): ~i32 {
    s | task: loop { on event => emit event } 
}
