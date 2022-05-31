# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
enum Shape[T] {
    Rectangle(T, T),
    Square(T),
}

def area(shape) = match shape {
    Shape::Rectangle(width, height) => width * height,
    Shape::Square(length) => 2 * length
}

def main() {
    val a0 = area(Shape::Rectangle(5, 3));
    val a1 = area(Shape::Square(3));
}
# ANCHOR_END: example
