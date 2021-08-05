extern type Cell(x: i32) {
    fun set(x: i32): unit;
    fun get(): i32;
}

fun main() {
    val c = Cell(5);
    val x = c.get();
    val y = c.set(x + x);
}
