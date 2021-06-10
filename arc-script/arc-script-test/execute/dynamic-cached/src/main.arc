fun pipe(s0: ~i32) -> ~i32 {
    val s1 = Identity() (s0);
    val s2 = Map(|x| x + 1) (s1);
    val s3 = Filter(|x| x != 0) (s2);
    val s4 = Duplicate() (s3);
    s4
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

task Duplicate() ~i32 -> ~i32 {
    on event => {
        emit event;
        emit event
    }
}

task Unique() ~i32 -> ~i32 {
    val set: Set[i32] = Set()
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
