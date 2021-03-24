task UniqueSet() ~i32 by i32 -> ~i32 by i32 {
    state set: {i32} = {}
    on event by key => {
        if event not in set {
            add set[event];
            emit event by key
        } else {
            unit
        }
    }
}

task UniqueMap() ~i32 by i32 -> ~i32 by i32 {
    state map: {i32 => unit} = {}
    on event by key => {
        if event not in map {
            map[event] = unit;
            emit event by key
        } else {
            unit
        }
    }
}
