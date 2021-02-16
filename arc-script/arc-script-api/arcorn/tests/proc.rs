use arcorn::conv;
use std::rc::Rc;

conv! {

    pub enum List {
        Cons(Rc<Cons>),
        Nil(Rc<Nil>),
    }

    pub struct Cons {
        pub val: i32,
        pub tail: Rc<List>,
    }

    pub struct Nil {}

    pub struct Foo {
        pub f0: Rc<String>,
        pub f1: i32,
        pub f2: i64,
        pub f3: u32,
        pub f4: u64,
    }

    pub enum Bar {
        A(Rc<String>),
        B(i32),
        C(i64),
        D(u32),
        E(u64),
    }
}

#[test]
fn test() {
    let tail = Rc::new(List::Nil(Rc::new(Nil {})));
    let tail = Rc::new(List::Cons(Rc::new(Cons { val: 0, tail })));
    let tail = Rc::new(List::Cons(Rc::new(Cons { val: 2, tail })));
    let tail = Rc::new(List::Cons(Rc::new(Cons { val: 3, tail })));
    let tail = Rc::new(List::Cons(Rc::new(Cons { val: 4, tail })));

    let arcon_tail: arcon_types::List = tail.as_ref().into();
    let _arc_tail: List = arcon_tail.into();
}

// To view generated code:
// $ cargo install cargo-expand
// $ cargo expand --tests codegen
