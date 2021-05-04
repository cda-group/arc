task MapWithState(init: i32, mapper: fun(i32, i32): -> (i32, i32)) ~i32 -> ~i32 {
    state value: i32 = init;
    on event => {
        let (new_value, new_event) = mapper(value, event) in
        value = new_value;
        emit new_event
    }
}
