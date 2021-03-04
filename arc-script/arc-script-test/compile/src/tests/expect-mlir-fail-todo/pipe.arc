fun foo(x: i32) -> i32 {
    x + x
}

fun bar(x: i32) -> i32 {
    x |> foo |> foo
}

task Baz() ~i32 -> ~i32 {
    on event => emit event
}

fun qux(s: ~i32) -> ~i32 {
    s |> Baz() |> Baz()
}
