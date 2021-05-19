#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

use arctime::macros::rewrite;

#[rewrite(handler = "handler")]
mod my_task {
    struct MyTask {
        state_variable: String,
    }
    enum InputPorts {
        A(i32),
        B(i32),
    }
    enum OutputPorts {
        C(i32),
        D(i32),
    }
}

impl MyTask {
    fn handler(&mut self, event: InputPorts) {
        match event {
            A(x) => self.emit(C(x)),
            B(x) => self.emit(D(x)),
        }
    }
}
