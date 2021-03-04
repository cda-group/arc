fun test1() -> i32 {

    let a = 3 in
    let b = |x| x + 1 in
    let c = |x| x - 1 in

    let d: i32 = a |> b |> c in
    # let d = c(b(a)) in

    d
}
