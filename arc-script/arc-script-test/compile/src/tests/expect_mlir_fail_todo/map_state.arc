task MapCount(): ~i32 by i32 -> ~i32 by i32 {
    state c: {i32 => i32} = {}
    on event by key => {
        c[event] = c[event] + 1;
        emit c[event] by key
    }
}
