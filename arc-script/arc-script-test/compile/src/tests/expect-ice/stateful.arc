task Stateful(init: i32) ~i32 -> ~i32 {
    state value: i32 = init;
    extern fun update(v: i32) -> i32;
    on event => emit update(event)
}
