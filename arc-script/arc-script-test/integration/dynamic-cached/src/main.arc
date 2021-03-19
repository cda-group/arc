fun pipe(s: ~i32) -> ~i32 {
    Identity() (s)
}

task Identity() ~i32 -> ~i32 {
    on event => emit event
}

task Map(f: fun(i32) -> i32) ~i32 -> ~i32 {
    on event => emit f(event)
}

task Filter(f: fun(i32) -> bool) ~i32 -> ~i32 {
    on event => if f(event) {
        emit event
    } else {
        unit
    }
}

task Unique() ~i32 -> ~i32 {
    state set: {i32} = {}
    on event => {
        if event not in set {
            add set[event];
            emit event
        } else {
            unit
        }
    }
}

task ExternTest() ~i32 -> ~i32 {
    extern fun rust_method(x: i32) -> i32
    fun arc_method(x: i32) -> i32 {
        x - 5
    }
    on event => emit rust_method(arc_method(event))
}
