task Null() () -> () { }

# -------------------------------

task Nat() () -> (Out(i32)) {
    port Loop(i32);
    emit Loop(0);
    on Loop(n) => {
        emit Out(n);
        emit Loop(n+1)
    }
}

# -------------------------------

task Clone(): i32 -> (A(i32), B(i32)) {
    on event => {
        emit A(event);
        emit B(event)
    }
}

# -------------------------------

task Merge(): (A(i32), B(i32)) -> i32 {
    on {
        A(event) => emit event,
        B(event) => emit event
    }
}

# -------------------------------

task Flip(): (A0(i32), B0(i32)) -> (A1(i32), B1(i32)) {
    on {
        A0(event) => emit B1(event),
        B0(event) => emit A1(event)
    }
}

# -------------------------------

task Split(f: fun(i32): bool): i32 -> (A(i32), B(i32)) {
    on event => if f(event) {
        emit A(event)
    } else {
        emit B(event)
    }
}

# -------------------------------

task Scan(i: i32, f: fun(i32, i32): i32): i32 -> i32 {
    var agg: i32 = i;
    on event => {
        agg = f(agg, event);
        emit agg
    }
}

# -------------------------------

task Until(f: fun(i32): bool): i32 -> i32 {
    on event => if f(event) {
        exit
    } else {
        emit event
    }
}

# -------------------------------

task Search(f: fun(i32): bool): i32 -> i32 {
    on event => if f(event) {
        emit event;
        exit
    } else {
        ()
    }
}

# -------------------------------

# Option 1:

task Fold(i: i32, f: fun(i32, i32) -> i32) (In(i32)) -> (Out($i32)) {
    state agg: i32 = i;
    on {
        In($) => exit Out(agg)
        In(event) => agg = f(agg, event),
    }
}

# Option 2:

task Fold(i: i32, f: fun(i32, i32) -> i32) (In(~i32)) -> (Out(i32)) {
    state agg: i32 = i;
    on {
        In($) => exit Out(agg)
        In(event) => agg = f(agg, event),
    }
}

# -------------------------------

# Problem: Must be able to both stream and evaluate into futures
# * Every stream can terminate which sends an End-of-Stream marker
#   * Should termination be encoded in the type or just be assumed?
# * When a task terminates, it might produce a value

task ScanFold(i: i32, f: fun(i32, i32) -> i32) i32 -> (Out(i32), $(i32)) {
    state agg: i32 = i;
    on {
        $ => exit agg
        event => {
            agg = f(agg, event);
            emit Out(agg)
        },
    }
}
