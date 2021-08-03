//! Macros for using structs.

/// Constructs a struct.
///
/// ```
/// use arc_script::arcorn;
/// mod foo {
///     use arc_script::arcorn;
///     #[arcorn::rewrite]
///     pub struct Bar {
///         x: i32,
///         y: i32
///     }
/// }
/// let x = arcorn::new!(Bar { x: i32, y: i32 });
/// ```
#[macro_export]
macro_rules! new {
    (@done $path:tt { $($arg:tt)* }) => {
        ($path { $($arg)* }).convert()
    };
    ($mod:tt :: $struct:tt { $($arg:tt)* }) => {
        arc_script::arcorn::paste!(arc_script::arcorn::new!(@done $mod::[<Concrete $struct>] { $($arg)* }))
    };
    ($struct:tt { $($arg:tt)* }) => {
        arc_script::arcorn::paste!(arc_script::arcorn::new!(@done [<Concrete $struct>] { $($arg)* }))
    };
}

/// Access a struct's field.
///
/// ```
/// use arc_script::arcorn;
/// mod foo {
///     use arc_script::arcorn;
///     #[arcorn::rewrite]
///     pub struct Bar {
///         x: i32,
///         y: i32
///     }
/// }
/// let a = arcorn::new!(Bar { x: i32, y: i32 });
/// let b = arcorn::access!(a, x);
/// ```
#[macro_export]
macro_rules! access {
    ($arg:expr, $field:tt) => {
        $arg.clone().$field.clone()
    };
}
