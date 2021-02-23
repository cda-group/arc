use std::rc::Rc;

pub fn callee_void_void() {
}

pub fn callee_si32_si32(x: i32) -> i32 {
    return x
}

pub fn callee_si32_x2_si32(a : i32, b : i32) -> i32 {
    return a + b
}

pub fn callee_tuple(x : Rc<(i32, i32, )>) -> Rc<(i32, i32, )> {
    return x
}


pub fn callee_struct(x : arctorustforeigncalls_types::ArcStructFfooTi32) -> arctorustforeigncalls_types::ArcStructFfooTi32 {
    return x
}

pub fn callee_mixed(x : Rc<(i32,i32, arctorustforeigncalls_types::ArcStructFaTi32)>)
		    -> Rc<(i32,i32, arctorustforeigncalls_types::ArcStructFaTi32)> {
    return x
}
