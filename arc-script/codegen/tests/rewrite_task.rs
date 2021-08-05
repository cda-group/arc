#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

use arc_script::codegen::*;

#[arc_script::codegen::rewrite(on_event = "handler", on_start = "startup")]
pub mod my_task {
    pub struct MyTask {
        pub param_variable: i32,
        #[state]
        pub state_variable: i32,
    }

    #[arc_script::codegen::rewrite]
    pub enum InputPorts {
        A(i32),
        B(i32),
    }

    #[arc_script::codegen::rewrite]
    pub enum OutputPorts {
        C(i32),
        D(i32),
    }
}

use my_task::*;

impl MyTask {
    pub fn handler(&mut self, event: InputPorts) {
        let x0 = is!(A, event.clone());
        if x0 {
            let x1: i32 = unwrap!(A, event.clone());
            let x2: OutputPorts = enwrap!(C, x1.clone());
            self.emit(x2.clone());
        } else {
            let x3 = is!(B, event.clone());
            if x3 {
                let x4: i32 = unwrap!(B, event.clone());
                let x5: OutputPorts = enwrap!(D, x4.clone());
                self.emit(x5.clone());
            }
        }
    }
    pub fn startup(&mut self) {
        self.state_variable.set(self.param_variable * 2)
    }
}
