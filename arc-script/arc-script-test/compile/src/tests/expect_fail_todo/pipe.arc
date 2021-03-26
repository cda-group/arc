fun test1(): i32 {

    let a: i32 = 3 in
    let b: fun(i32): i32 = fun(x): x + 1 in
    let c: fun(i32): i32 = fun(x): x - 1 in

    let d: i32 = a |> b |> c in
    # let d = c(b(a)) in

    d
}

fun test2(): i32 {

    let a: i32 = 3 in
    let b: fun(i32, i32): (i32, i32) = fun((x, y)): (y, x) in
    let c: fun(i32): i32 = fun(x): x + 1 in

    let d: (i32, i32) = (a, a) |> b |> (c, c) in
    # let (x0, x1) = b(a, a) in
    # let d = (c(x0), c(x1)) in

    d
}
