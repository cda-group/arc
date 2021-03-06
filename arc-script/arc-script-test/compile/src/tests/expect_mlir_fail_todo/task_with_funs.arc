task Test(x: i32) ~i32 -> ~i32 {
    fun addx(y: i32) -> i32 {
        let z = x + y in
        z
    }
    on event => {
        emit addx(event)
    }
}
