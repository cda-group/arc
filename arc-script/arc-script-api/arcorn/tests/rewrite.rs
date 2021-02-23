use arc_script::arcorn;

#[arcorn::rewrite]
struct Point {
    x: i32,
    y: i32,
}

#[arcorn::rewrite]
enum Foo {
    Bar(i32),
    Baz(f32)
}
