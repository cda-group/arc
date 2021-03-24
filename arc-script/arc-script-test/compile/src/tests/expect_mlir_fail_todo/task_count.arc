task Count() ~i32 by i32 -> ~i32 by i32 {
    state count: i32 = 0
    on event by key => {
        count = count + 1;
        emit count by key
    }
}
