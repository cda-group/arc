extern type Cell(x: i32) {
    fun get(): i32;
    fun set(x: i32): unit;
}

task Stateful(init: i32): ~i32 by i32 -> ~i32 by i32 {
    val state: crate::Cell[i32] = crate::Cell(init);
    extern fun update(v: i32): i32;
    on event by key => emit update(state.get()) by key;
}
