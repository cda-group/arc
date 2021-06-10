fun pipe(s0: ~i32) -> ~i32 {
    val s1 = Identity() (s0);
    val s2 = Map(|x| x + 1) (s1);
    val s3 = Filter(|x| x != 0) (s2);
    val s4 = Duplicate() (s3);
    s4
}

# Operators currently do not support polymorphism

task Identity() ~i32 -> ~i32 {
    on event => emit event
}

task Map(mapper: fun(i32) -> i32) ~i32 -> ~i32 {
    on event => emit mapper(event)
}

task Filter(predicate: fun(i32) -> bool) ~i32 -> ~i32 {
    on event => if predicate(event) {
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
