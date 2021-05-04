task MinuteSum(): In(~i32) -> Out(~i32) {
    port Trigger
    state sum: i32 = 0
    emit Trigger after 10s
    on {
        In(event) => {
            sum = sum + event
        },
        Trigger => {
            emit Out(sum);
            sum = 0;
            emit Trigger after 10s
        }
    }
}
