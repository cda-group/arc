/// Get the value of a variable.
///
/// ```
/// use arc_runtime::prelude::*;
/// let a = 5;
/// let b = val!(a);
/// ```
#[cfg(feature = "legacy")]
#[macro_export]
macro_rules! val {
    ($arg:expr) => {
        $arg.clone()
    };
}

#[cfg(not(feature = "legacy"))]
#[macro_export]
macro_rules! val {
    ($arg:expr) => {
        $arg
    };
}

macro_rules! inline {
    ($($tt:tt)*) => { $($tt)* };
}

/// Access a struct's field.
///
/// ```
/// use arc_runtime::prelude::*;
/// #[rewrite]
/// pub struct Bar {
///     pub x: i32,
///     pub y: i32
/// }
/// #[rewrite(main)]
/// fn main() {
///     let a = new!(Bar { x: 0, y: 1 });
///     let b = access!(a, x);
/// }
/// ```
#[macro_export]
macro_rules! access {
    ($arg:expr, $field:tt) => {
        $arg.clone().$field.clone()
    };
}

#[macro_export]
macro_rules! letroot {
    ($var_name:ident : $t:ty  = $stack:expr, $value:expr) => {
        let stack: &ShadowStack = &$stack;
        let value = $value;
        #[allow(unused_unsafe)]
        let mut $var_name = unsafe {
            ShadowStackInternal::<$t>::construct(
                stack,
                stack.head.get(),
                core::mem::transmute::<_, TraitObject>(&value as &dyn Rootable).vtable as usize,
                value,
            )
        };
        #[allow(unused_unsafe)]
        stack
            .head
            .set(unsafe { core::mem::transmute(&mut $var_name) });
        #[allow(unused_mut)]
        let mut $var_name = unsafe { Rooted::construct(&mut $var_name.value) };
    };

    ($var_name:ident = $stack:expr, $value:expr) => {
        let stack: &ShadowStack = &$stack;
        let value = $value;
        #[allow(unused_unsafe)]
        let mut $var_name = unsafe {
            ShadowStackInternal::<_>::construct(
                stack,
                stack.head.get(),
                core::mem::transmute::<_, TraitObject>(&value as &dyn Rootable).vtable as usize,
                value,
            )
        };
        #[allow(unused_unsafe)]
        stack
            .head
            .set(unsafe { core::mem::transmute(&mut $var_name) });
        #[allow(unused_mut)]
        #[allow(unused_unsafe)]
        let mut $var_name = unsafe { Rooted::construct(&mut $var_name.value) };
    };
}

#[macro_export]
macro_rules! _vector {
    ([$($x:expr),* $(,)?], $ctx:expr) => {{
        let stack = $ctx.mutator().shadow_stack();
        letroot!(vec = stack, Some(Vec::new($ctx)));

        $(
            vec.as_mut().unwrap().0.push($ctx.mutator(), $x);
            vec.as_mut().unwrap().0.write_barrier($ctx.mutator());
        )*
        vec.take().unwrap()
    }}
}
