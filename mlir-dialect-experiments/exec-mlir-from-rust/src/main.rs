#[link(name = "hello")]
extern "C" {
    fn my_function(x: i32) -> i32;
}

fn main() {
    unsafe {
        println!("my_function(1) == {}", my_function(1));
    }
}
