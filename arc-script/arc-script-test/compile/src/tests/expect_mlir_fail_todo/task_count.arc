task Count() ~i32 -> ~i32 {
    state count: i32 = 0
    on event => {
        count = count + 1;
        emit count
    }
}
