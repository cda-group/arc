pub trait Stream<T> {
    fn operator(&self);
}

#[no_mangle]
pub extern "C" fn pipeline(input: Box<dyn Stream<i32>>) -> Box<dyn Stream<i32>> {
    input
}
