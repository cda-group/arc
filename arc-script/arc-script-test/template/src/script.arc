fun pipe(event: ~i32) -> ~i32 {
    Identity() (event)
}

task Identity() ~i32 -> ~i32 {
    on event => emit event
}
