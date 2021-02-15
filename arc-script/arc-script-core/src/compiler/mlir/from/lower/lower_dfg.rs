use crate::compiler::dfg::from::eval::value::ValueKind;
use crate::compiler::dfg::DFG;
use crate::compiler::hir;

use crate::compiler::mlir;
use crate::compiler::mlir::ConstKind;

use arc_script_core_shared::Lower;

use super::Context;

impl Lower<ConstKind, Context<'_>> for ValueKind {
    #[rustfmt::skip]
    fn lower(&self, _ctx: &mut Context<'_>) -> ConstKind {
        match self {
            Self::Unit          => ConstKind::Unit,
            Self::I8(v)         => ConstKind::I8(*v),
            Self::I16(v)        => ConstKind::I16(*v),
            Self::I32(v)        => ConstKind::I32(*v),
            Self::I64(v)        => ConstKind::I64(*v),
            Self::U8(v)         => ConstKind::U8(*v),
            Self::U16(v)        => ConstKind::U16(*v),
            Self::U32(v)        => ConstKind::U32(*v),
            Self::U64(v)        => ConstKind::U64(*v),
            Self::Bf16(v)       => ConstKind::Bf16(*v),
            Self::F16(v)        => ConstKind::F16(*v),
            Self::F32(v)        => ConstKind::F32(*v),
            Self::F64(v)        => ConstKind::F64(*v),
            Self::Char(v)       => ConstKind::Char(*v),
            Self::Str(_)        => todo!(),
            Self::Bool(_v)       => todo!(),
            Self::Item(_)       => todo!(),
            Self::Task(_, _)    => todo!(),
            Self::Stream(_, _)  => todo!(),
            Self::Vector(_)     => todo!(),
            Self::Tuple(_)      => todo!(),
            Self::Array(_)      => todo!(),
            Self::Variant(_, _) => todo!(),
            Self::Struct(_)     => todo!(),
        }
    }
}

impl Lower<mlir::Fun, Context<'_>> for DFG {
    fn lower(&self, ctx: &mut Context<'_>) -> mlir::Fun {
        let mut ops = Vec::new();
        for node in self.graph.node_indices() {
            let node_name = ctx.info.names.intern(format!("node_{}", node.index()));
            let node = self.graph.node_weight(node).unwrap();
            let (vars, mut args): (Vec<_>, Vec<_>) = node
                .frame
                .iter()
                .map(|(&x, v)| {
                    let kind = mlir::OpKind::Const(v.kind.lower(ctx));
                    let var = mlir::Var::new(x.into(), v.ty);
                    (var, mlir::Op::new(var.into(), kind, None))
                })
                .unzip();
            let node_ty = ctx.info.types.intern(hir::TypeKind::Nominal(node.path));
            ops.append(&mut args);
            ops.push(mlir::Op::new(
                mlir::Var::new(node_name.into(), node_ty).into(),
                mlir::OpKind::Node(node.path, vars),
                None,
            ));
        }
        for edge in self.graph.edge_indices() {
            let (origin, target) = self.graph.edge_endpoints(edge).unwrap();

            let origin_name = ctx.info.names.intern(format!("node_{}", origin.index()));
            let target_name = ctx.info.names.intern(format!("node_{}", target.index()));

            let edge = self.graph.edge_weight(edge).unwrap();
            let origin = self.graph.node_weight(origin).unwrap();
            let target = self.graph.node_weight(target).unwrap();

            let origin_ty = ctx.info.types.intern(hir::TypeKind::Nominal(origin.path));
            let target_ty = ctx.info.types.intern(hir::TypeKind::Nominal(target.path));

            let origin_var = mlir::Var::new(origin_name.into(), origin_ty);
            let target_var = mlir::Var::new(target_name.into(), target_ty);

            ops.push(mlir::Op::new(
                None,
                mlir::OpKind::Edge((origin_var, edge.oport), (target_var, edge.iport)),
                None,
            ));
        }

        let main = ctx.info.names.intern("main").into();
        let main = ctx.info.paths.intern_child(ctx.info.paths.root, main).into();
        let vars = vec![];
        let body = mlir::Region::new(vec![mlir::Block::new(ops)]);
        let ty = ctx.info.types.intern(hir::ScalarKind::Unit);
        mlir::Fun::new(main, vars, body, ty)
    }
}
