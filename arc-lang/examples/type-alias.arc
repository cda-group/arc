# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

extern def sqrt(Num): Num;

# ANCHOR: example
type Num = i32;
type Point = #{x:Num, y:Num};
type Line = #{start:Point, end:Point};

def length(line) = {
    val a = line.start.x - line.end.x;
    val b = line.start.y - line.end.y;
    sqrt(a**2 + b**2)
}

def main() = {
    val p0 = #{x:0, y:1};
    val p1 = #{x:5, y:9};
    val line = #{start:p0, end:p1};
    length(line);
}
# ANCHOR_END: example
