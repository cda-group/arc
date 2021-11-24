//! Macros for using structs.

/// Constructs a struct.
///
/// ```
/// mod foo {
///     #[arc_codegen::rewrite]
///     pub struct Bar {
///         x: i32,
///         y: i32
///     }
/// }
/// let x = arc_codegen::new!(Bar { x: i32, y: i32 });
/// ```
#[macro_export]
macro_rules! new {
    (@done $path:tt { $($arg:tt)* }) => {
        ($path { $($arg)* }).convert()
    };
    ($mod:tt :: $struct:tt { $($arg:tt)* }) => {
        arc_codegen::paste!(arc_codegen::new!(@done $mod::[<Concrete $struct>] { $($arg)* }))
    };
    ($struct:tt { $($arg:tt)* }) => {
        arc_codegen::paste!(arc_codegen::new!(@done [<Concrete $struct>] { $($arg)* }))
    };
}

/// Access a struct's field.
///
/// ```
/// mod foo {
///     #[arc_codegen::rewrite]
///     pub struct Bar {
///         x: i32,
///         y: i32
///     }
/// }
/// let a = arc_codegen::new!(Bar { x: i32, y: i32 });
/// let b = arc_codegen::access!(a, x);
/// ```
#[macro_export]
macro_rules! access {
    ($arg:expr, $field:tt) => {
        $arg.clone().$field.clone()
    };
}
