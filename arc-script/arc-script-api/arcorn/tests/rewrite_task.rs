#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

#[arc_script::arcorn::rewrite(on_event = "handler", on_start = "startup")]
pub mod my_task {
    pub struct MyTask {
        pub param_variable: i32,
        #[state]
        pub state_variable: i32,
    }
    #[arc_script::arcorn::rewrite]
    pub enum InputPorts {
        A(i32),
        B(i32),
    }
    #[arc_script::arcorn::rewrite]
    pub enum OutputPorts {
        C(i32),
        D(i32),
    }
}

use my_task::*;

impl MyTask {
    pub fn handler(&mut self, event: InputPorts) {
        match event.this.unwrap() {
            A(x) => self.emit(C(x).wrap()),
            B(x) => self.emit(D(x).wrap()),
        }
    }
    pub fn startup(&mut self) {
        self.state_variable.set(self.param_variable * 2)
    }
}
