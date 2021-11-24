pub trait Stream<T> {
    fn operator(&self);
}

impl<T> Stream<T> for ConcreteStream<T> {
    fn operator(&self) {}
}

pub struct ConcreteStream<T> {
    elems: Vec<T>,
}

fn call_dynamic() -> Result<u32, Box<dyn std::error::Error>> {
    unsafe {
        let lib = libloading::Library::new("/path/to/liblibrary.so")?;
        let func: libloading::Symbol<unsafe extern "C" fn() -> u32> = lib.get(b"my_func")?;
        Ok(func())
    }
}

fn main() {
    println!("Hello, world!");
}
