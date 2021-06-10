extern type Cell(x: i32) {
    fun get(): i32;
    fun set(x: i32): unit;
}

task TumblingWindowSum(): ~i32 by i32 -> ~i32 by i32 {
    val agg: crate::Cell = crate::Cell(0);

    every 1m {
        emit agg.get() by 0;
        agg.set(0);
    };

    on event by key => agg.set(agg.get() + event);
}
