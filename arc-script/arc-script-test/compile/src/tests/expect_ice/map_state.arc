task MapCount(): ~i32 by i32 -> ~i32 by i32 {
    val c: Dict[i32, i32] = Dict();
    on event by key => {
        c[event] = c[event] + 1;
        emit c[event] by key
    };
}
