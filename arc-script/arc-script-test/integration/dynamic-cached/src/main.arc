fun pipe(s: ~i32) -> ~i32 {
    Identity() (s)
}

task Identity() (i32) -> (i32) {
    on event => emit event
}
