# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

enum Expr {
    Num(i32),
    Add((i32, i32))
}

def eval(e) {
    match e {
        Expr::Num(x) => x,
        Expr::Add((a, b)) => a + b,
        _ => 0
    }
}
