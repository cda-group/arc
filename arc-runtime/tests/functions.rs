#![allow(dead_code)]

use arc_runtime::prelude::*;

#[rewrite]
fn foo(x: i32) {
    let a: i32 = x - 1;
    if x == 0 {
        println!("Hello, world!");
    } else {
        foo((a,))
    }
}

// Expands into:

// fn _foo(x: i32, ctx: Context) {
//     let a: i32 = x - 1;
//     if x == 0 {
//         println!("Hello, world!");
//     } else {
//         _foo(a, ctx)
//     }
// }

#[rewrite(main)]
fn main() {
    let x: String = String::from_str("Hello, world!");
    let y: &str = "Hello, world!";
    let _z: unit = String::push_str(x, y);
}

// Expands into:

// fn _bar() {
//     let system = &KompactConfig::default().build().unwrap();
//     let mutator = &mut instantiate_immix(ImmixOptions::default());
//     let ctx = Context::new(system, mutator);
//
//     let stack: &ShadowStack = &ctx.mutator.shadow_stack();
//     let value = String::from_str("Hello, world!", ctx);
//     #[allow(unused_unsafe)]
//     let mut x = unsafe {
//         ShadowStackInternal::<String>::construct(
//             stack,
//             stack.head.get(),
//             core::mem::transmute::<_, TraitObject>(&value as &dyn Rootable).vtable as usize,
//             value,
//         )
//     };
//     #[allow(unused_unsafe)]
//     stack.head.set(unsafe { core::mem::transmute(&mut x) });
//     #[allow(unused_mut)]
//     let mut x = unsafe { Rooted::construct(&mut x.value) };
//     let y: &str = "Hello, world!";
//     let _z: unit = String::push_str(x.clone(), y, ctx);
// }
