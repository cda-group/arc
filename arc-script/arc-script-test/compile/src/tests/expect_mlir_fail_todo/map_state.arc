task MapCount() ~i32 -> ~i32 {
    state c: {i32 => i32} = {}
    on event => {
        c[event] = c[event] + 1;
        emit c[event]
    }
}
