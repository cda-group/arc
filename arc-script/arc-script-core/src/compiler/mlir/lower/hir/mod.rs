use crate::compiler::hir;
use crate::compiler::hir::utils::SortFields;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::mlir;

use arc_script_core_shared::get;
use arc_script_core_shared::lower;
use arc_script_core_shared::Lower;
use arc_script_core_shared::New;
use arc_script_core_shared::Shrinkwrap;
use arc_script_core_shared::VecDeque;
use arc_script_core_shared::VecMap;

#[derive(New, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Context<'i> {
    hir: &'i HIR,
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i mut Info,
    pub(crate) ops: Vec<mlir::Op>,
}

lower! {
    [node, ctx, hir]

    hir::Item => mlir::Item {
        mlir::Item::new(node.kind.lower(ctx), node.loc)
    },
    hir::ItemKind => mlir::ItemKind {
        match node {
            hir::ItemKind::Fun(item)         => mlir::ItemKind::Fun(item.lower(ctx)),
            hir::ItemKind::Enum(item)        => mlir::ItemKind::Enum(item.lower(ctx)),
            hir::ItemKind::Task(_item)       => todo!(),
            hir::ItemKind::ExternFun(_item)  => todo!(),
            hir::ItemKind::ExternType(_item) => todo!(),
            hir::ItemKind::TypeAlias(_)      => unreachable!(),
            hir::ItemKind::Variant(_)        => unreachable!(),
        }
    },
    hir::Fun => mlir::Fun {
        mlir::Fun {
            path: node.path,
            params: node.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>(),
            body: node.body.lower(mlir::OpKind::Return, ctx),
            t: node.body.var.t,
        }
    },
    hir::Enum => mlir::Enum {
        let variants = node
            .variants
            .iter()
            .map(|path| get!(&ctx.hir.resolve(path).kind, hir::ItemKind::Variant(x)).lower(ctx))
            .collect::<Vec<_>>();
        mlir::Enum::new(node.path, variants)
    },
    hir::Variant => mlir::Variant {
        mlir::Variant::new(node.path, node.t, node.loc)
    },
    hir::Param => mlir::Var {
        match &node.kind {
            hir::ParamKind::Ignore => todo!(),
            hir::ParamKind::Ok(x) => mlir::Var::new(*x, node.t),
            hir::ParamKind::Err => unreachable!(),
        }
    },
    hir::BinOp => mlir::BinOp {
        let kind = match &node.kind {
            hir::BinOpKind::Add  => mlir::BinOpKind::Add,
            hir::BinOpKind::Sub  => mlir::BinOpKind::Sub,
            hir::BinOpKind::Mul  => mlir::BinOpKind::Mul,
            hir::BinOpKind::Div  => mlir::BinOpKind::Div,
            hir::BinOpKind::Mod  => mlir::BinOpKind::Mod,
            hir::BinOpKind::Pow  => mlir::BinOpKind::Pow,
            hir::BinOpKind::Equ  => mlir::BinOpKind::Equ,
            hir::BinOpKind::Neq  => mlir::BinOpKind::Neq,
            hir::BinOpKind::Or   => mlir::BinOpKind::Or,
            hir::BinOpKind::And  => mlir::BinOpKind::And,
            hir::BinOpKind::Xor  => mlir::BinOpKind::Xor,
            hir::BinOpKind::Band => mlir::BinOpKind::Band,
            hir::BinOpKind::Bor  => mlir::BinOpKind::Bor,
            hir::BinOpKind::Bxor => mlir::BinOpKind::Bxor,
            hir::BinOpKind::Gt   => mlir::BinOpKind::Gt,
            hir::BinOpKind::Lt   => mlir::BinOpKind::Lt,
            hir::BinOpKind::Geq  => mlir::BinOpKind::Geq,
            hir::BinOpKind::Leq  => mlir::BinOpKind::Leq,
            hir::BinOpKind::Mut  => mlir::BinOpKind::Mut,
            hir::BinOpKind::In   => crate::todo!(),
            hir::BinOpKind::Err  => unreachable!(),
        };
        mlir::BinOp::new(kind)
    },
    hir::LitKind => mlir::ConstKind {
        match node {
            hir::LitKind::I8(v)       => mlir::ConstKind::I8(*v),
            hir::LitKind::I16(v)      => mlir::ConstKind::I16(*v),
            hir::LitKind::I32(v)      => mlir::ConstKind::I32(*v),
            hir::LitKind::I64(v)      => mlir::ConstKind::I64(*v),
            hir::LitKind::U8(v)       => mlir::ConstKind::U8(*v),
            hir::LitKind::U16(v)      => mlir::ConstKind::U16(*v),
            hir::LitKind::U32(v)      => mlir::ConstKind::U32(*v),
            hir::LitKind::U64(v)      => mlir::ConstKind::U64(*v),
            hir::LitKind::Bf16(v)     => mlir::ConstKind::Bf16(*v),
            hir::LitKind::F16(v)      => mlir::ConstKind::F16(*v),
            hir::LitKind::F32(v)      => mlir::ConstKind::F32(*v),
            hir::LitKind::F64(v)      => mlir::ConstKind::F64(*v),
            hir::LitKind::Bool(v)     => mlir::ConstKind::Bool(*v),
            hir::LitKind::Char(v)     => mlir::ConstKind::Char(*v),
            hir::LitKind::Str(_)      => todo!(),
            hir::LitKind::DateTime(_) => todo!(),
            hir::LitKind::Duration(_) => todo!(),
            hir::LitKind::Unit        => mlir::ConstKind::Unit,
            hir::LitKind::Err         => unreachable!(),
        }
    },
    Vec<hir::Var> => Vec<mlir::Var> {
        node.iter().map(|v| v.lower(ctx)).collect()
    },
    VecMap<hir::Name, hir::Var> => VecMap<mlir::Name, mlir::Var> {
        // NOTE: We sort because name-mangling is order-sensitive with respect to fields.
        // Because the arguments to the struct are variables, their order cannot produce
        // side-effects.
        node.into_iter().map(|(x, v)| (*x, v.lower(ctx))).collect::<VecMap<_, _>>().sort_fields(ctx.info)
    },
    VecDeque<hir::Stmt> => Vec<mlir::Op> {
        node.iter().map(|stmt| stmt.lower(ctx)).collect()
    },
    hir::Stmt => mlir::Op {
        match &node.kind {
            hir::StmtKind::Assign(i) => mlir::Op::new(Some(i.param.lower(ctx)), i.expr.lower(ctx), node.loc),
        }
    },
    hir::Var => mlir::Var {
        if let hir::VarKind::Ok(x, _) = &node.kind {
            mlir::Var::new(*x, node.t)
        } else {
            unreachable!()
        }
    },
    hir::Expr => mlir::OpKind {
        match ctx.hir.exprs.resolve(node.id) {
            hir::ExprKind::Return(v)        => mlir::OpKind::Return(v.lower(ctx)),
            hir::ExprKind::Break(v)         => mlir::OpKind::Break(v.lower(ctx)),
            hir::ExprKind::Continue         => mlir::OpKind::Continue,
            hir::ExprKind::Item(x)          => match ctx.hir.resolve(x).kind {
                hir::ItemKind::Fun(_)       => mlir::OpKind::Const(mlir::ConstKind::Fun(*x)),
                hir::ItemKind::Task(_)      => crate::todo!(),
                _ => unreachable!(),
            },
            hir::ExprKind::Call(v, vs)       => mlir::OpKind::CallIndirect(v.lower(ctx), vs.lower(ctx)),
            hir::ExprKind::SelfCall(x, vs)   => crate::todo!(),
            hir::ExprKind::Invoke(v, x, vs)  => mlir::OpKind::CallMethod(v.lower(ctx), *x, vs.lower(ctx)),
            hir::ExprKind::Select(_e, _es)   => todo!(),
            hir::ExprKind::Lit(l)            => mlir::OpKind::Const(l.lower(ctx)),
            hir::ExprKind::BinOp(v0, op, v1) => mlir::OpKind::BinOp(v0.t, v0.lower(ctx), op.lower(ctx), v1.lower(ctx)),
            hir::ExprKind::UnOp(op, v)       => mlir::OpKind::UnOp(mlir::UnOp::new(op.kind.lower(ctx)), v.lower(ctx)) ,
            hir::ExprKind::Access(v, x)      => mlir::OpKind::Access(v.lower(ctx), *x),
            hir::ExprKind::Project(v, i)     => mlir::OpKind::Project(v.lower(ctx), i.id),
            hir::ExprKind::If(v, b0, b1)     => mlir::OpKind::If(
                v.lower(ctx),
                b0.lower(mlir::OpKind::Result, ctx),
                b1.lower(mlir::OpKind::Result, ctx)
            ),
            hir::ExprKind::Array(vs)         => mlir::OpKind::Array(vs.lower(ctx)),
            hir::ExprKind::Struct(fs)        => mlir::OpKind::Struct(fs.lower(ctx)),
            hir::ExprKind::Enwrap(x, v1)     => mlir::OpKind::Enwrap(*x, v1.lower(ctx)),
            hir::ExprKind::Unwrap(x, v1)     => mlir::OpKind::Unwrap(*x, v1.lower(ctx)),
            hir::ExprKind::Is(x, v1)         => mlir::OpKind::Is(*x, v1.lower(ctx)),
            hir::ExprKind::Tuple(vs)         => mlir::OpKind::Tuple(vs.lower(ctx)),
            hir::ExprKind::Emit(v)           => mlir::OpKind::Emit(v.lower(ctx)),
            hir::ExprKind::Log(v)            => mlir::OpKind::Log(v.lower(ctx)),
            hir::ExprKind::Loop(v)           => crate::todo!(),
            hir::ExprKind::After(_, _)       => crate::todo!(),
            hir::ExprKind::Every(_, _)       => crate::todo!(),
            hir::ExprKind::Cast(_, _)        => crate::todo!(),
            hir::ExprKind::Unreachable       => crate::todo!(),
            hir::ExprKind::Initialise(x, v)  => crate::todo!(),
            hir::ExprKind::Err               => unreachable!(),
        }
    },
    hir::UnOpKind => mlir::UnOpKind {
        match node {
            hir::UnOpKind::Not => mlir::UnOpKind::Not,
            hir::UnOpKind::Neg => mlir::UnOpKind::Neg,
            hir::UnOpKind::Err => unreachable!(),
        }
    },
}

impl hir::Block {
    fn lower(
        &self,
        terminator: impl FnOnce(mlir::Var) -> mlir::OpKind,
        ctx: &mut Context<'_>,
    ) -> mlir::Block {
        let mut ops = self.stmts.lower(ctx);
        ops.push(mlir::Op::new(
            None,
            terminator(self.var.lower(ctx)),
            self.var.loc,
        ));
        mlir::Block::new(ops)
    }
}
