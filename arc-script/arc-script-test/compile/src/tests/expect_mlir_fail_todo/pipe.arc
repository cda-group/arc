fun foo(x: i32): i32 {
    x + x
}

fun bar(x: i32): i32 {
    x |> foo |> foo
}

task Baz(): ~i32 by i32 -> ~i32 by i32 {
    on event by key => emit event by key
}

fun qux(s: ~i32 by i32): ~i32 by i32 {
    s |> Baz() |> Baz()
}
