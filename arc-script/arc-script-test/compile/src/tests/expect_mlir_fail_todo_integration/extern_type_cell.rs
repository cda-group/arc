use std::cell::RefCell;
use std::rc::Rc;

mod script {
    arc_script::include!("src/tests/expect_mlir_fail_todo/extern_type_cell.rs");
    type Cell = super::Cell<i32>;
}

#[derive(Clone)]
struct Cell<T> {
    interior: Rc<RefCell<T>>,
}

impl<T: Clone> Cell<T> {
    fn new(x: T) -> Self {
        Self {
            interior: Rc::new(RefCell::new(x)),
        }
    }
    fn get(self) -> T {
        self.interior.borrow().clone()
    }
    fn set(self, x: T) {
        *self.interior.borrow_mut() = x;
    }
}
