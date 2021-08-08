fun fib(x: i32): i32 {
    if x > 1 {
        fib(x-1) + fib(x-2)
    } else {
        x
    }
}
