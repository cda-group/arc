task Test(x: i32) ~i32 -> ~i32 {
    fun add(y: i32) -> i32 {
        let z = x + y in
        z
    }
    on event => {
      let x = add(event) in
      emit x
    }
}
