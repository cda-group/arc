fun pipe(s0: ~i32) -> ~i32 {
    let s1 = Identity() (s0) in
    let s2 = Map(|x| x + 1) (s1) in
    let s3 = Filter(|x| x != 0) (s2) in
    let s4 = Duplicate() (s3) in
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
