#![allow(unused)]

#[allow(non_camel_case_types)]
#[cfg(test)]
mod test_toplevel {

    use arc_runtime::prelude::*;

    declare_functions!(f);

    #[rewrite]
    fn f(a: i32) -> i32 {
        a + a
    }

    #[rewrite(main)]
    #[test]
    fn test() {
        let x: function!((i32) -> i32) = function!(f);
        let y: function!((i32,) -> i32) = function!(f);
        let y: i32 = call_indirect!(x(1));
        let y: i32 = call_indirect!(x(1,));
        let z: i32 = call!(f(1));
        let z: i32 = call!(f(1,));
    }
}
