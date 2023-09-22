use crate::Value;
use builtins::path::Path;
use hir::Type;
use im_rc::Vector;
use value::dynamic::Dataflow;
use value::dynamic::Instance;
use value::dynamic::Sink;
use value::dynamic::Stream;
use value::dynamic::StreamKind;
use value::ValueKind::VUnit;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("run", |ctx, t, v| {
            let v0 = v[0].as_dataflow();
            let ss = hir_reachable::process(&mut ctx.ctx7, v0.clone(), ctx.ss.clone());
            let is = hir_to_rust::process(&mut ctx.ctx9, ss);
            let package = ctx.ctx10.new_package().unwrap();
            let mut ctx = codegen::Context::file(package.main.clone());
            ctx.colors(false);
            ctx.writeln(&is, write_rust::write).unwrap();
            ctx.writeln(&v0, write_rust::write_dataflow).unwrap();
            package.build().unwrap().run().unwrap();
            Instance {
                log: Path::from(package.root.join("log")),
            }
            .into()
        })
        .f("merge", |ctx, t, v| {
            let v0 = v[0].as_array().0;
            let streams = v0
                .clone()
                .into_iter()
                .flat_map(|x| x.as_dataflow().streams)
                .collect();
            let sinks = v0.into_iter().flat_map(|x| x.as_dataflow().sinks).collect();
            Dataflow::new(streams, sinks).into()
        });
}

// pub fn reachable(&mut self, c: Component) -> Vector<hir::Stmt> {
//     let ss = hir_reachable::process(&mut self.ctx6.ctx7, c, self.ctx5.stmts.clone());
//     if self.config.show.reachable {
//         if !ss.is_empty() {
//             self.show_hir(&ss);
//         }
//         return Vector::new();
//     } else {
//         ss
//     }
// }
//
// pub fn hir_to_mlir(&mut self, c: Component, ss: Vector<hir::Stmt>) -> Vector<mlir::Item> {
//     hir_to_mlir::process(&mut self.ctx6.ctx8, c, ss)
// }
//
// pub fn hir_to_rust(&mut self, ss: Vector<hir::Stmt>) -> Vector<rust::Item> {
//     hir_to_rust::process(&mut self.ctx6.ctx9, ss)
// }

// pub fn run(&mut self, c: Component, is: Vector<rust::Item>) -> Result<()> {
//     let package = self.ctx6.ctx10.new_package()?;
//     let mut ctx = codegen::Context::file(package.main.clone());
//     ctx.colors(false);
//     ctx.writeln(&is, write_rust::write)?;
//     ctx.writeln(&c, write_rust::write_component)?;
//     package.build()?.run()?;
//     Ok(())
// }

// pub fn components(&self) -> Vec<Component> {
//     self.ctx6.dataflow.components()
// }

// pub fn list_dataflows(&self) -> Result<(), std::io::Error> {
//     self.components()
//         .iter()
//         .try_for_each(|component| write_hir_value::print_component(component))
// }

// pub fn codegen(&mut self, items: Vector<mlir::Item>) -> Result<(), std::io::Error> {
//     if !self.has_errors() {
//         // codegen::Context::println(&items, mlir_to_mlir::pr_module)?;
//     } else {
//         self.emit_errors();
//     }
//     Ok(())
// }
