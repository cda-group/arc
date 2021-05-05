task MinuteSum(): ~i32 -> ~i32 {
    every 1m {
        emit sum
    };
    on event => sum = sum + event;
}
