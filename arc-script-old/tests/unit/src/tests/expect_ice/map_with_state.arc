task MapWithState(init: i32, mapper: fun(i32, i32): (i32, i32)): ~i32 -> ~i32 {
    val state: Cell[i32] = Cell(init);
    on event => {
        val (new_value, new_event) = mapper(state.get(), event);
        state.set(new_value);
        emit new_event
    }
}
