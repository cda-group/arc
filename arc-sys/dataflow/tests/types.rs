mod basic1 {
    use dataflow::prelude::*;

    #[rewrite]
    struct Point {
        x: i32,
        y: i32,
    }

    #[rewrite]
    enum Foo {
        Bar(i32),
        Baz(f32),
    }
}

mod basic2 {
    use dataflow::prelude::*;

    #[rewrite]
    struct A {
        b: B,
    }

    #[rewrite]
    struct B {
        c: i32,
    }
}

mod basic3 {
    use dataflow::prelude::*;

    #[rewrite]
    enum A {
        B(B),
        C(C),
    }

    #[rewrite]
    struct B {
        v: i32,
    }

    #[rewrite]
    struct C {}
}

mod list {
    use dataflow::prelude::*;

    #[rewrite]
    enum List {
        Cons(Cons),
        Nil(unit),
    }

    #[rewrite]
    struct Cons {
        v: i32,
        t: List,
    }

    #[test]
    fn test() {
        let l = enwrap!(List::Nil, unit);
        let _x = is!(List::Cons, l);
        let h = new!(Cons { v: 5, t: l });
        let l = enwrap!(List::Cons, h);
        let h = unwrap!(List::Cons, l);
        assert_eq!(h.v, 5);
    }
}

mod structs {
    use dataflow::prelude::*;

    #[rewrite]
    struct Foo {
        a: i32,
        b: Bar,
    }

    #[rewrite]
    struct Bar {}

    #[test]
    fn test() {
        let x0 = new!(Bar {});
        let _f = new!(Foo { a: 0, b: x0 });
    }
}

mod unit {
    use dataflow::prelude::*;

    #[rewrite]
    enum Foo {
        Bar(unit),
    }

    #[test]
    fn test() {
        let _x = enwrap!(Foo::Bar, unit);
    }
}

mod compact_structs {
    use dataflow::prelude::*;

    #[rewrite(compact)]
    struct CompactFoo {
        a: i32,
        b: CompactBar,
    }

    #[rewrite(compact)]
    struct CompactBar {}

    #[test]
    fn test() {
        let x0 = new!(CompactBar {});
        let _f = new!(CompactFoo { a: 0, b: x0 });
    }
}

mod compact_enums {
    use dataflow::prelude::*;

    #[rewrite(compact)]
    enum CompactFoo {
        Bar(i32),
        Baz(CompactBaz),
    }

    #[rewrite(compact)]
    enum CompactBaz {
        Qux(i32),
    }

    #[test]
    fn test() {
        enwrap!(CompactFoo::Bar, 1);
        enwrap!(CompactFoo::Baz, enwrap!(CompactBaz::Qux, 2));
    }
}
