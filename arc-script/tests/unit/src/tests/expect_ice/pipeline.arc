task Identity(): (Input(i32)) -> (Output(i32)) {
    on Input(event) => emit Output(event);
}

fun main(x0: ~i32): ~i32 {
    val x1 = Identity() (x0);
    val x2 = Identity() (x1);
    val x3 = Identity() (x2);
    val x4 = Identity() (x3);
    val x5 = Identity() (x4);
    val x6 = Identity() (x5);
    val x7 = Identity() (x6);
    val x8 = Identity() (x7);
    val x9 = Identity() (x8);
    x9
}
