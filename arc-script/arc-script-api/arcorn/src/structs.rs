// /// Accesses a field of a struct.
// ///
// /// ```
// /// use arc_script::arcorn;
// ///
// /// #[arcorn::rewrite]
// /// #[derive(Copy, Eq, PartialEq)]
// /// struct Foo {
// ///     a: i32,
// ///     b: Bar,
// /// }
// ///
// /// #[arcorn::rewrite]
// /// #[derive(Copy, Eq, PartialEq)]
// /// struct Bar {}
// ///
// /// let foo = Foo::new(3, Bar::new());
// /// let a = arcorn::access!(foo, a);
// /// let b = arcorn::access!(foo, b);
// /// assert_eq!(a, 3);
// /// assert_eq!(b, Bar::new());
// /// ```
// #[macro_export]
// macro_rules! access {
//     {
//         $struct:expr , $member:ident
//     } => {
//         $struct.$member()
//     }
// }
