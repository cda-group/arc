# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def area(shape) = shape.x * shape.y

def main() {
    val line = #{x:5};
    val rect = #{y:10|line};
    val cube = #{z:20|rect};

    area(line); # ERROR
    area(rect); # OK
    area(cube); # OK
}
# ANCHOR_END: example
