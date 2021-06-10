
task Foo(): i32 -> (Even(i32), Odd(i32)) {
    on event => {
        val output = if x % 2 == 0 {
            Even(event)
        } else {
            Odd(event)
        };
        emit output
    };
}
