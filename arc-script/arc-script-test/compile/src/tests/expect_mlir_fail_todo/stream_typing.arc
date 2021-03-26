fun test0(x: ~i32 by i32): ~i32 by i32 {
    x |> test1
}

# fun test0[K](x: ~i32 by K): ~i32 by K {
#     x |> test1
# }

fun test1(x: ~i32 by i32): ~i32 by i32 {
    x
}
