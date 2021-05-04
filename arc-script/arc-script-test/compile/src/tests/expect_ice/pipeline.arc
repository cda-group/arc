task Identity(): (Input(i32)) -> (Output(i32)) {
    on Input(event) => emit Output(event)
}

fun main(x0: ~i32): ~i32 {
    let x1 = Identity() (x0) in
    let x2 = Identity() (x1) in
    let x3 = Identity() (x2) in
    let x4 = Identity() (x3) in
    let x5 = Identity() (x4) in
    let x6 = Identity() (x5) in
    let x7 = Identity() (x6) in
    let x8 = Identity() (x7) in
    let x9 = Identity() (x8) in
    x9
}
