# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
val a = { { { { { 6 } } } } };
val b = { 1; 2; 3; 4; 5; 6 };
val c = { 1; { 2; { 3; { 4; { 5; { 6 } } } } } };
val d = { { { { { { 1 }; 2 }; 3 }; 4 }; 5 }; 6 };
val e = a + b + c + d;
# ANCHOR_END: example
}
