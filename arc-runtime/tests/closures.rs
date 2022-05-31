// TODO: `serde_state` does not support type erasure.
//
// #![allow(unused)]
// #[allow(non_camel_case_types)]
// mod test_closure {
//
//     use arc_runtime::prelude::*;
//
//     declare_functions!(f);
//
//     #[rewrite]
//     pub struct Closure {
//         pub fun: Function<(i32, Erased), i32>,
//         pub env: Erased,
//     }
//
//     #[rewrite(erase)]
//     pub struct Env {
//         pub b: i32,
//     }
//
//     #[rewrite]
//     fn f(a: i32, env: Erased) -> i32 {
//         let env: Env = unerase!(env, Env);
//         let b: i32 = env.b;
//         a + b
//     }
//
//     #[rewrite(main)]
//     #[test]
//     fn main() {
//         let x: function!((i32, Erased) -> i32) = function!(f);
//         let env: Env = new!(Env { b: 1 });
//         let env: Erased = erase!(env, Env);
//         let y: i32 = call_indirect!(x(1, env));
//         let y: i32 = call_indirect!(x(1, env,));
//     }
// }

#[allow(non_camel_case_types)]
#[cfg(test)]
mod test_toplevel {

    use arc_runtime::prelude::*;

    declare!(functions: [f], tasks: []);

    #[rewrite]
    fn f(a: i32) -> i32 {
        a + a
    }

    #[rewrite(main)]
    #[test]
    fn test() {
        let x0: function!((i32) -> i32) = function!(f);
        let x1: function!((i32,) -> i32) = function!(f);
        let x2: i32 = call_indirect!(x0(1));
        let x3: i32 = call_indirect!(x1(1,));
        let x4: i32 = call!(f(1));
        let _x5: i32 = call!(f(1,));
    }
}
