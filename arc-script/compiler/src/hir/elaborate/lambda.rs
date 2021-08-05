//! Lambda lifting. https://gist.github.com/jozefg/652f1d7407b7f0266ae9
//!
//! The idea of lambda lifting convert all closures into top-level functions.
//!
//! The following issues must be addressed:
//! 1. Closures may capture variables.
//!
//!    Top-level functions may only capture global variables.
//!    The rest need to be given as parameters.
//!
//! 2. Closures do not have a name.
//!
//!    Top-level functions must be addressed by name.
//!
//! 3. Closures are values which can be bound to variables and propagated
//!    throughout the program.
//!
//!    Any code which uses the closure must be augmented.
//!
//! The algorithm has the following steps:
//! 1. Find each closure.
//! 2. Calculate the set of free variables in its body.
//!   * This is the set of occurring variables modulo parameters and globals.
//! 3. Generate the environment struct containing all variables.
//! 3. Augment all variable accesses into fi
//! 3. Convert the closure into a pair
//!
//! ==== Before =================
//! fun foo(a: i32, b: i32)
//!   let lam0 = |x:i32| x + a;
//!   let lam1 = |y:i32| y + b;
//!   bar(lam0);
//!   bar(lam1);
//! end
//!
//! fun bar(lam: i32 -> i32)
//!   lam(5)
//! end
//! =============================
//!
//! In the static-dispatch version of the algorithm, the output is:
//!
//! ==== After (Static) =========
//! fun f0(x:i32, env:{a:i32})
//!   x + env.a
//! end
//!
//! fun f1(x:i32, env:{b:i32})
//!   y + env.b
//! end
//!
//! fun foo(a: i32, b: i32)
//!   let lam0 = {f: f0, env: {a=a}};
//!   let lam1 = {f: f1, env: {b=b}};
//!   bar(lam0);
//!   bar(lam1);
//! end
//!
//! fun bar0(lam: {f: fn_ptr(i32 -> i32), env: {a:i32}})
//!   lam.f(5, lam.env)
//! end
//!
//! fun bar1(lam: {f: fn_ptr(i32 -> i32), env: {a:i32}})
//!   lam.f(5, lam.env)
//! end
//! =============================

use super::Context;
use crate::ast;
use crate::hir;
use crate::info::names::NameId;
use crate::shared::Set;
use crate::shared::VecMap;

pub(super) fn lower(params: &Vec<ast::Pat>, body: ast::Expr, ctx: Context) -> hir::Expr {
    //     let fv = body.free_vars();
    crate::todo!()
}

