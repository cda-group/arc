fun pipe(x: ~i32) -> ~i32 {
    Identity() (x)
}

task Identity() ~i32 -> ~i32 {
    on event => emit event
}

