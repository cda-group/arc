#![allow(unused)]

mod basic1 {
    #[arc_script::arcorn::rewrite]
    pub struct Point {
        pub x: i32,
        pub y: i32,
    }

    #[arc_script::arcorn::rewrite]
    pub enum Foo {
        Bar(i32),
        Baz(f32),
    }
}

mod basic2 {
    #[arc_script::arcorn::rewrite]
    pub struct A {
        pub b: B,
    }

    #[arc_script::arcorn::rewrite]
    pub struct B {
        pub c: i32,
    }
}

mod basic3 {
    #[arc_script::arcorn::rewrite]
    pub enum A {
        B(B),
        C(C),
    }

    #[arc_script::arcorn::rewrite]
    pub struct B {
        pub v: i32,
    }

    #[arc_script::arcorn::rewrite]
    pub struct C {}
}

mod list {
    #[arc_script::arcorn::rewrite]
    pub enum List {
        Cons(Cons),
        Nil(Nil),
    }

    #[arc_script::arcorn::rewrite]
    pub struct Cons {
        pub val: i32,
        pub tail: Box<List>,
    }

    #[arc_script::arcorn::rewrite]
    pub struct Nil {}

    #[test]
    fn test() {
        let l = arc_script::arcorn::enwrap!(Nil, Nil::new());
        let l = arc_script::arcorn::enwrap!(Cons, Cons::new(5, Box::new(l)));
        let h = arc_script::arcorn::unwrap!(Cons, l);
        assert_eq!(h.val, 5);
    }
}

mod structs {
    #[arc_script::arcorn::rewrite]
    #[derive(Eq, PartialEq)]
    pub struct Foo {
        pub a: i32,
        pub b: Bar,
    }

    #[arc_script::arcorn::rewrite]
    #[derive(Eq, PartialEq)]
    pub struct Bar {}

    #[test]
    fn test() {
        Foo { a: 0, b: Bar {} };
    }
}

mod unit {
    #[arc_script::arcorn::rewrite]
    #[derive(Eq, PartialEq)]
    pub enum Foo {
        A(()),
    }

    #[test]
    fn test() {
        arc_script::arcorn::enwrap!(A, ());
    }
}
