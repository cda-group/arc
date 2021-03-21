task KeyBy(keyfun: fun(i32) -> i32) ~i32 by i32 -> ~i32 by i32 {
    on event by key => emit event by keyfun(event)
}
