use crate::hir;
use crate::hir::utils::SortFields;
use crate::info::Info;
use crate::mlir;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::lower;
use arc_script_compiler_shared::Lower;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecDeque;
use arc_script_compiler_shared::VecMap;

#[derive(New, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Context<'i> {
    hir: &'i hir::HIR,
    pub(crate) mlir: &'i mut mlir::MLIR,
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i mut Info,
    pub(crate) ops: Vec<mlir::Op>,
}

lower! {
    [node, ctx, hir]

    hir::HIR => () {
        for x in &node.namespace {
            let item = node.resolve(x).lower(ctx);
            ctx.mlir.intern(*x, item);
            ctx.mlir.items.push(*x);
        }
    },
    hir::Item => mlir::Item {
        mlir::Item::new(node.kind.lower(ctx), node.loc)
    },
    hir::ItemKind => mlir::ItemKind {
        match node {
            hir::ItemKind::Fun(item)        => mlir::ItemKind::Fun(item.lower(ctx)),
            hir::ItemKind::Enum(item)       => mlir::ItemKind::Enum(item.lower(ctx)),
            hir::ItemKind::Task(item)       => mlir::ItemKind::Task(item.lower(ctx)),
            hir::ItemKind::ExternFun(item)  => mlir::ItemKind::ExternFun(item.lower(ctx)),
            hir::ItemKind::ExternType(item) => mlir::ItemKind::ExternType(item.lower(ctx)),
            hir::ItemKind::TypeAlias(_)     => unreachable!(),
            hir::ItemKind::Variant(_)       => unreachable!(),
        }
    },
    hir::ExternFun => mlir::ExternFun {
        mlir::ExternFun {
            path: node.path,
            params: node.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>(),
            rt: node.rt,
        }
    },
    hir::ExternType => mlir::ExternType {
        mlir::ExternType {
            path: node.path,
            params: node.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>(),
            items: node.items.iter().map(|p| {
                let item = ctx.hir.resolve(p).lower(ctx);
                ctx.mlir.intern(p, item);
                *p
            }).collect::<Vec<_>>(),
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
    hir::Task => mlir::Task {
        let on_event = get!(&ctx.hir.resolve(node.on_event.fun).kind, hir::ItemKind::Fun(x));
        let on_start = get!(&ctx.hir.resolve(node.on_start.fun).kind, hir::ItemKind::Fun(x));
        let fields = node.params.iter().map(|p| {
            let x = get!(p.kind, hir::ParamKind::Ok(x));
            (x, p.t)
        }).collect();
        let params = node.params.iter().map(|p| p.lower(ctx)).collect();
        let this_t = ctx.types.intern(hir::TypeKind::Struct(fields));
        let ievent = on_event.params.first().unwrap().lower(ctx);
        // This is mangled later on
        let oevent_t = ctx.types.intern(hir::TypeKind::Nominal(node.ointerface.interior.id.into()));
        let on_event = on_event.body.lower(mlir::OpKind::Return, ctx);
        let on_start = on_start.body.lower(mlir::OpKind::Return, ctx);
        let iinterior_enum_item = ctx.hir.resolve(node.iinterface.interior).lower(ctx);
        let ointerior_enum_item = ctx.hir.resolve(node.ointerface.interior).lower(ctx);
        let istream_ts = node.iinterface.exterior.clone();
        let ostream_ts = node.ointerface.exterior.clone();
        ctx.mlir.intern(node.iinterface.interior, iinterior_enum_item);
        ctx.mlir.intern(node.ointerface.interior, ointerior_enum_item);
        ctx.mlir.items.push(*node.iinterface.interior);
        ctx.mlir.items.push(*node.ointerface.interior);
        for x in &node.namespace {
            let item = ctx.hir.resolve(x).lower(ctx);
            ctx.mlir.intern(x, item);
            ctx.mlir.items.push(*x);
        }
        mlir::Task {
            path: node.path,
            params,
            istream_ts,
            ostream_ts,
            this_t,
            ievent,
            oevent_t,
            on_event,
            on_start,
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
    hir::Param => mlir::Param {
        let kind = match &node.kind {
            hir::ParamKind::Ignore => mlir::VarKind::Ok(ctx.names.fresh()),
            hir::ParamKind::Ok(x) => if node.t.is_unit(ctx.info) {
                mlir::VarKind::Elided
            } else {
                mlir::VarKind::Ok(*x)
            },
            hir::ParamKind::Err => unreachable!(),
        };
        mlir::Param::new(kind, node.t)
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
            hir::LitKind::F32(v)      => mlir::ConstKind::F32(*v),
            hir::LitKind::F64(v)      => mlir::ConstKind::F64(*v),
            hir::LitKind::Bool(v)     => mlir::ConstKind::Bool(*v),
            hir::LitKind::Char(v)     => mlir::ConstKind::Char(*v),
            hir::LitKind::Str(_)      => todo!(),
            hir::LitKind::DateTime(_) => todo!(),
            hir::LitKind::Duration(_) => todo!(),
            hir::LitKind::Unit        => mlir::ConstKind::Noop,
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
            hir::StmtKind::Assign(a) => {
                mlir::Op::new(a.param.lower(ctx), a.expr.lower(ctx), node.loc)
            },
        }
    },
    hir::Var => mlir::Var {
        let (x, scope) = get!(node.kind, hir::VarKind::Ok(x, scope));
        let kind = if node.t.is_unit(ctx.info) {
            mlir::VarKind::Elided
        } else {
            mlir::VarKind::Ok(x)
        };
        mlir::Var::new(kind, scope, node.t)
    },
    hir::Expr => mlir::OpKind {
        match ctx.hir.exprs.resolve(node.id) {
            hir::ExprKind::Return(v)        => mlir::OpKind::Return(v.lower(ctx)),
            hir::ExprKind::Break(v)         => mlir::OpKind::Break(v.lower(ctx)),
            hir::ExprKind::Continue         => mlir::OpKind::Continue,
            hir::ExprKind::Item(x)          => match ctx.hir.resolve(x).kind {
                hir::ItemKind::Fun(_)       => mlir::OpKind::Const(mlir::ConstKind::Fun(*x)),
                hir::ItemKind::Task(_)      => mlir::OpKind::Const(mlir::ConstKind::Fun(*x)),
                _ => unreachable!(),
            },
            hir::ExprKind::Call(v, vs)       => mlir::OpKind::CallIndirect(v.lower(ctx), vs.lower(ctx)),
            hir::ExprKind::SelfCall(x, vs)   => crate::todo!(),
            hir::ExprKind::Invoke(v, x, vs)  => mlir::OpKind::CallMethod(v.lower(ctx), *x, vs.lower(ctx)),
            hir::ExprKind::Select(_e, _es)   => todo!(),
            hir::ExprKind::Lit(l)            => mlir::OpKind::Const(l.lower(ctx)),
            hir::ExprKind::BinOp(v0, op, v1) => mlir::OpKind::BinOp(v0.lower(ctx), op.lower(ctx), v1.lower(ctx)),
            hir::ExprKind::UnOp(op, v)       => mlir::OpKind::UnOp(mlir::UnOp::new(op.kind.lower(ctx)), v.lower(ctx)) ,
            hir::ExprKind::Access(v, x)      => if node.t.is_unit(ctx.info) {
                mlir::OpKind::Noop
            } else {
                mlir::OpKind::Access(v.lower(ctx), *x)
            },
            hir::ExprKind::Project(v, i)     => mlir::OpKind::Project(v.lower(ctx), i.id),
            hir::ExprKind::If(v, b0, b1)     => mlir::OpKind::If(
                v.lower(ctx),
                b0.lower(mlir::OpKind::Result, ctx),
                b1.lower(mlir::OpKind::Result, ctx)
            ),
            hir::ExprKind::Array(vs)         => mlir::OpKind::Array(vs.lower(ctx)),
            hir::ExprKind::Struct(fs)        => mlir::OpKind::Struct(fs.lower(ctx)),
            hir::ExprKind::Enwrap(x, v1)     => mlir::OpKind::Enwrap(*x, v1.lower(ctx)),
            hir::ExprKind::Unwrap(x, v1)     => if node.t.is_unit(ctx.info) {
                mlir::OpKind::Noop
            } else {
                mlir::OpKind::Unwrap(*x, v1.lower(ctx))
            },
            hir::ExprKind::Is(x, v1)         => mlir::OpKind::Is(*x, v1.lower(ctx)),
            hir::ExprKind::Tuple(vs)         => mlir::OpKind::Tuple(vs.lower(ctx)),
            hir::ExprKind::Emit(v)           => mlir::OpKind::Emit(v.lower(ctx)),
            hir::ExprKind::Log(v)            => mlir::OpKind::Log(v.lower(ctx)),
            hir::ExprKind::Loop(v)           => crate::todo!(),
            hir::ExprKind::After(_, _)       => crate::todo!(),
            hir::ExprKind::Every(_, _)       => crate::todo!(),
            hir::ExprKind::Cast(_, _)        => crate::todo!(),
            hir::ExprKind::Unreachable       => mlir::OpKind::Panic,
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
            mlir::Param::new(mlir::VarKind::Elided, self.var.t),
            terminator(self.var.lower(ctx)),
            self.var.loc,
        ));
        mlir::Block::new(ops)
    }
}
