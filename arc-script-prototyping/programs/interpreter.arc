enum Expr {
    Num(i32),
    Add((i32, i32))
}

fun eval(e) {
    match e {
        Expr::Num(x) => x,
        Expr::Add((a, b)) => a + b,
        _ => 0
    }
}