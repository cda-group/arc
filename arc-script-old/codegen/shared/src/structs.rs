//! Macros for using structs.

/// Constructs a struct.
///
/// ```
/// use arc_script::codegen;
/// mod foo {
///     use arc_script::codegen;
///     #[codegen::rewrite]
///     pub struct Bar {
///         x: i32,
///         y: i32
///     }
/// }
/// let x = codegen::new!(Bar { x: i32, y: i32 });
/// ```
#[macro_export]
macro_rules! new {
    (@done $path:tt { $($arg:tt)* }) => {
        ($path { $($arg)* }).convert()
    };
    ($mod:tt :: $struct:tt { $($arg:tt)* }) => {
        arc_script::codegen::paste!(arc_script::codegen::new!(@done $mod::[<Concrete $struct>] { $($arg)* }))
    };
    ($struct:tt { $($arg:tt)* }) => {
        arc_script::codegen::paste!(arc_script::codegen::new!(@done [<Concrete $struct>] { $($arg)* }))
    };
}

/// Access a struct's field.
///
/// ```
/// use arc_script::codegen;
/// mod foo {
///     use arc_script::codegen;
///     #[codegen::rewrite]
///     pub struct Bar {
///         x: i32,
///         y: i32
///     }
/// }
/// let a = codegen::new!(Bar { x: i32, y: i32 });
/// let b = codegen::access!(a, x);
/// ```
#[macro_export]
macro_rules! access {
    ($arg:expr, $field:tt) => {
        $arg.clone().$field.clone()
    };
}
