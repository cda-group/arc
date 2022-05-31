# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

enum Expr {
    Num(i32),
    Add(Expr, Expr)
}

def eval(e) = match e {
    Expr::Num(x) => x,
    Expr::Add(a, b) => eval(a) + eval(b),
}

def main() {
    val x = eval(Expr::Num(1));
    val y = eval(Expr::Add(1, 2));
    assert(x == 1);
    assert(y == 3);
}
