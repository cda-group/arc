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
