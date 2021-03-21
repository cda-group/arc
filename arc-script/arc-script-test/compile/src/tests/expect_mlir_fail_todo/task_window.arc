task TumblingWindowSum() ~i32 by i32 -> ~i32 by i32 {
    state agg: i32 = 0

    timer unit

    startup trigger unit by 0 after 1m

    timeout unit by key => {
        emit agg by key;
        agg = 0;
        trigger unit by key after 1m
    }

    on event by key => agg = agg + event
}
