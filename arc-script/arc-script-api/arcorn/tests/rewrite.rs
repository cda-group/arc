// mod basic1 {
//     use arc_script::arcorn;
//
//     #[arcorn::rewrite]
//     struct Point {
//         x: i32,
//         y: i32,
//     }
//
//     #[arcorn::rewrite]
//     enum Foo {
//         Bar(i32),
//         Baz(f32),
//     }
//
//     #[test]
//     fn test() {
//         let p = Point::new(1, 2);
//         //         let foo = arcorn::enwrap!(Bar, 3);
//     }
// }
//
// mod basic2 {
//     use arc_script::arcorn;
//
//     #[arcorn::rewrite]
//     struct A {
//         b: B,
//     }
//
//     #[arcorn::rewrite]
//     struct B {
//         c: i32,
//     }
// }
//
// mod basic3 {
//     use arc_script::arcorn;
//
//     #[arcorn::rewrite]
//     enum A {
//         B(B),
//         C(C),
//     }
//
//     #[arcorn::rewrite]
//     struct B {
//         v: i32,
//     }
//
//     #[arcorn::rewrite]
//     struct C {}
// }
//
// mod list {
//     use arc_script::arcorn;
//
//     #[arcorn::rewrite]
//     enum List {
//         Cons(Cons),
//         Nil(Nil),
//     }
//
//     #[arcorn::rewrite]
//     struct Cons {
//         val: i32,
//         tail: Box<List>,
//     }
//
//     #[arcorn::rewrite]
//     struct Nil {}
//
//     #[test]
//     fn test() {
//         let list = List!(@enwrap, Cons, Cons::new(5, Box::new(List!(@enwrap, Nil, Nil::new()))));
//         let cons = List!(@unwrap, Cons, list);
//         let val: i32 = Cons!(@get, cons, val);
//         let nil: Box<List> = Cons!(@get, cons, tail);
//         assert_eq!(val, 5);
//     }
//
//     //     #[test]
//     //     fn test() {
//     //         let list = List!(
//     //             @enwrap,
//     //             Cons,
//     //             Cons::new(5, Box::new(List!(@enwrap, Nil, Nil::new())))
//     //         );
//     //
//     //         let cons = List!(@unwrap, Cons, list);
//     //         let val: i32 = Cons!(@get, cons, val);
//     //         let nil: Box<List> = Cons!(@get, cons, tail);
//     //         assert_eq!(val, 5);
//     //     }
// }
//
// mod structs {
//     use arc_script::arcorn;
//     #[arcorn::rewrite]
//     #[derive(Eq, PartialEq)]
//     struct Foo {
//         a: i32,
//         b: Bar,
//     }
//
//     #[arcorn::rewrite]
//     #[derive(Eq, PartialEq)]
//     struct Bar {}
//
//     #[test]
//     fn test() {
//         let foo = Foo::new(3, Bar::new());
//         // Notice how this works even though Bar is not Copy
//         assert_eq!(Foo!(@get, foo, b), Bar::new());
//         assert_eq!(Foo!(@get, foo, a), 3);
//     }
// }
//
// mod unit {
//     use arc_script::arcorn;
//     #[arcorn::rewrite]
//     #[derive(Eq, PartialEq)]
//     enum Foo {
//         A(()),
//     }
//
//     #[test]
//     fn test() {
//         let foo = arcorn::enwrap!(A, arcorn::Unit::new());
//     }
// }
