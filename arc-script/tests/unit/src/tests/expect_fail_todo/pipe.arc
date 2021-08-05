fun test1(): i32 {

    val a: i32 = 3;
    val b: fun(i32): i32 = fun(x): x + 1;
    val c: fun(i32): i32 = fun(x): x - 1;

    val d: i32 = a | b | c;
    # val d = c(b(a));

    d
}

fun test2(): i32 {

    val a: i32 = 3;
    val b: fun(i32, i32): (i32, i32) = fun((x, y)): (y, x);
    val c: fun(i32): i32 = fun(x): x + 1;

    val d: (i32, i32) = (a, a) | b | (c, c);
    # val (x0, x1) = b(a, a);
    # val d = (c(x0), c(x1));

    d
}
