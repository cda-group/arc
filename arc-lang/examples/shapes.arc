# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
enum Shape[T] {
    Rectangle(T, T),
    Circle(T),
}

def area(shape) = match shape {
    Shape::Rectangle(width, height) => width * height,
    Shape::Circle(radius) => 3.14 * radius ** 2
}

def main() {
    val a0 = area(Shape::Rectangle(5.0, 3.0));
    val a1 = area(Shape::Circle(3.0));
}
# ANCHOR_END: example
