
pub trait Stream<T> {
    fn operator(&self);
}
#[repr(C)]
pub struct AnimalVTable {
    speak: extern "C" fn speak(VRef<AnimalVTable>, i32, & mut SharedString);
}

#[no_mangle]
pub extern "C" fn pipeline(input: Box<dyn Stream<i32>>) -> Box<dyn Stream<i32>> {
    input
}
