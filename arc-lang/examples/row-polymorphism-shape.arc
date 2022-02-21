# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def area(shape) = shape.x * shape.y

def test() {
    val line = #{x:5};
    val rect = #{y:10|line};
    val cube = #{z:20|rect};

    area(line); # ERROR
    area(rect); # OK
    area(cube); # OK
}
# ANCHOR_END: example
