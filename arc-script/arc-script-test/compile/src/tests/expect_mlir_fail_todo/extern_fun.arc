extern fun increment(x: i32): i32;

task Adder(): ~i32 by i32 -> ~i32 by i32 {
    extern fun addition(x: i32, y: i32): i32;

    on event by key => emit addition(event, event) by key;
}

fun pipe(s: ~i32 by i32): ~i32 by i32 {
    if increment(1) == 2 {
        s | Adder()
    } else {
        s | Adder()
    }
}
