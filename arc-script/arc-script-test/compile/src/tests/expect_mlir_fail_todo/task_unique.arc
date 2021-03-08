task UniqueSet() ~i32 -> ~i32 {
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

task UniqueMap() ~i32 -> ~i32 {
    state map: {i32 => bool} = {}
    on event => {
        if event not in map {
            map[event] = true;
            emit event
        } else {
            unit
        }
    }
}
