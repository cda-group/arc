use crate::prelude::*;

impl SyntaxTree {
    // Post-order traversal (Leaves first)
    pub fn for_each_expr<F: FnMut(&mut Expr)>(&mut self, ref mut f: F) {
        self.for_each_fun(|fun| fun.body.for_each_expr(f));
        self.body.for_each_expr(f);
    }
    // Pre-order traversal (Parents first)
    pub fn for_each_expr_postorder<F: FnMut(&mut Expr)>(&mut self, ref mut f: F) {
        self.for_each_fun(|fun| fun.body.for_each_expr_postorder(f));
        self.body.for_each_expr_postorder(f)
    }

    pub fn for_each_fun<F: FnMut(&mut FunDef)>(&mut self, ref mut f: F) {
        self.fundefs.iter_mut().for_each(|(_, fundef)| f(fundef));
    }

    pub fn for_each_decl<F: FnMut(&mut Decl)>(&mut self, ref mut f: F, table: &mut SymbolTable) {
        table.decls.iter_mut().for_each(f);
    }

    pub fn for_each_task<F: FnMut(&mut TaskDef)>(&mut self, ref mut f: F) {
        self.taskdefs.iter_mut().for_each(|(_, taskdef)| f(taskdef));
    }
}

impl TaskDef {
    pub fn for_each_fun<F: FnMut(&mut FunDef)>(&mut self, ref mut f: F) {
        self.fundefs.iter_mut().for_each(|(_, fundef)| f(fundef));
    }
}

/// Macro for generating pre- and post-order visitors of expressions.
macro_rules! for_each_expr {
    {
        name: $name:ident,
        pre: $pre:literal,
        post: $post:literal
    } => {
        impl Expr {
            pub fn $name<F: FnMut(&mut Expr)>(&mut self, f: &mut F) {
                if $pre {
                    f(self);
                }
                match &mut self.kind {
                    If(c, t, e) => {
                        c.$name(f);
                        t.$name(f);
                        e.$name(f);
                    }
                    For(_, e, c, b) => {
                        e.$name(f);
                        if let Some(c) = c {
                            c.$name(f);
                        }
                        b.$name(f);
                    }
                    Match(e, cases) => {
                        e.$name(f);
                        cases.iter_mut().for_each(|(_, e)| e.$name(f));
                    }
                    Let(_, e1, e2) => {
                        e1.$name(f);
                        e2.$name(f);
                    },
                    ConsArray(ps) => ps.iter_mut().for_each(|p| p.$name(f)),
                    ConsTuple(ps) => ps.iter_mut().for_each(|p| p.$name(f)),
                    ConsStruct(fs) => fs.iter_mut().for_each(|(_, v)| v.$name(f)),
                    ConsVariant(_, arg) => arg.$name(f),
                    Lit(_) => {}
                    Var(_) => {}
                    BinOp(l, _, r) => {
                        l.$name(f);
                        r.$name(f);
                    }
                    UnOp(op, e) => {
                        match op {
                            Call(ps) => ps.iter_mut().for_each(|p| p.$name(f)),
                            _ => {}
                        }
                        e.$name(f);
                    }
                    Loop(cond, body) => {
                        cond.$name(f);
                        body.$name(f);
                    }
                    Closure(_, body) => body.$name(f),
                    ExprErr => {}
                }
                if $post {
                    f(self);
                }
            }
        }
    }
}

for_each_expr! {
    name: for_each_expr,
    pre: true,
    post: false
}

for_each_expr! {
    name: for_each_expr_postorder,
    pre: false,
    post: true
}

pub fn merge<T>(mut a: Vec<T>, mut b: Vec<T>) -> Vec<T> {
    a.append(&mut b);
    a
}
