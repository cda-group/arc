task Test(x: i32): ~i32 by i32 -> ~i32 by i32 {
    fun addx(y: i32): i32 {
        val z = x + y;
        z
    }
    on event by key => emit addx(event) by key;
}
