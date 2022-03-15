# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# Lambdas can both capture their environment and be passed around as values.

def main() {
    val a = 1;
    val b = 2;
    val x = fun(c) = a + b + c;
    val y = fun(b) = a + b;
    run(x);
    run(y);
}

# If a function takes a lambda as parameter, then it is polymorphic over the
# lambda's environment.

def run(lambda) = lambda(3)
# ANCHOR_END: example
