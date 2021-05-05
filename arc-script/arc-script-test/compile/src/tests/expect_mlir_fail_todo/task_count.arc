extern type Cell(x: i32) {
    fun get(): i32;
    fun set(x: i32): unit;
}

task Count(): ~i32 by i32 -> ~i32 by i32 {
    val count = crate::Cell(0);
    on event by key => {
        count.set(count.get() + 1);
        emit count.get() by key
    };
}
