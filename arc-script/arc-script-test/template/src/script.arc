fun pipe(event: ~i32 by unit) -> ~i32 by unit {
    event |> Identity()
}

task Identity() ~i32 by unit -> ~i32 by unit {
    on event => emit event
}
