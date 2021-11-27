task Identity(): i32 -> i32 {
    loop {
        on event => emit event
    }
}
