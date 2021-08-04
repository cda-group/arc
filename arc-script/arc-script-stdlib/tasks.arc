task Id(): ~i32 by i32 -> ~i32 by i32 {
    on event by key => emit event by key;
}

# -------------------------------

task Map(f: fun(i32): i32): ~i32 by i32 -> ~i32 by i32 {
    on event by key => emit f(event) by key;
}

# -------------------------------

task Filter(f: fun(i32): bool): ~i32 by i32 -> ~i32 by i32 {
    on event by key => if f(event) {
        emit event by key
    };
}
