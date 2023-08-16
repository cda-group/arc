#![allow(unused)]

use anyhow::Result;
use config::Config;
use diagnostics::Error;
use hir::Stmt;
use hir::Type;
use im_rc::Vector;
use logging::Logger;
use sources::Sources;
use value::Value;

#[derive(Debug)]
pub struct Compiler {
    pub config: Config,
    pub(crate) ctx0: parser::context::Context,
    pub(crate) ctx1: ast_to_hir::context::Context,
    pub(crate) ctx2: hir_lambda_lift::context::Context,
    pub(crate) ctx3: hir_type_inference::context::Context,
    pub(crate) ctx4: hir_patcomp::context::Context,
    pub(crate) ctx5: hir_monomorphise::context::Context,
    pub(crate) ctx6: hir_interpreter::context::Context,
}

impl Compiler {
    pub fn init(config: Config) -> Self {
        Self {
            config,
            ctx0: Default::default(),
            ctx1: Default::default(),
            ctx2: Default::default(),
            ctx3: Default::default(),
            ctx4: Default::default(),
            ctx5: Default::default(),
            ctx6: Default::default(),
        }
    }

    pub fn new(config: Config, logger: Logger) -> Self {
        let mut this = Self::init(config);
        this.ctx0.diagnostics.backtrace = this.config.show.backtrace;
        this.ctx1.diagnostics.backtrace = this.config.show.backtrace;
        this.ctx2.diagnostics.backtrace = this.config.show.backtrace;
        this.ctx3.diagnostics.backtrace = this.config.show.backtrace;
        this.ctx4.diagnostics.backtrace = this.config.show.backtrace;
        this.ctx5.diagnostics.backtrace = this.config.show.backtrace;
        this.ctx6.ctx7.diagnostics.backtrace = this.config.show.backtrace;
        this.ctx6.ctx8.diagnostics.backtrace = this.config.show.backtrace;
        this.ctx6.ctx9.diagnostics.backtrace = this.config.show.backtrace;

        this.ctx0.diagnostics.failfast = this.config.failfast;
        this.ctx1.diagnostics.failfast = this.config.failfast;
        this.ctx2.diagnostics.failfast = this.config.failfast;
        this.ctx3.diagnostics.failfast = this.config.failfast;
        this.ctx4.diagnostics.failfast = this.config.failfast;
        this.ctx5.diagnostics.failfast = this.config.failfast;
        this.ctx6.ctx7.diagnostics.failfast = this.config.failfast;
        this.ctx6.ctx8.diagnostics.failfast = this.config.failfast;
        this.ctx6.ctx9.diagnostics.failfast = this.config.failfast;
        this
    }

    pub fn sources(&mut self) -> &mut Sources {
        &mut self.ctx0.sources
    }

    // pub fn dataflow(&mut self) -> &mut Dataflow {
    //     &mut self.ctx6.dataflow
    // }

    pub fn parse(
        &mut self,
        name: impl Into<String>,
        source: impl Into<String>,
    ) -> Vector<ast::Stmt> {
        let ss = parser::parse_program(&mut self.ctx0, name, source);
        if self.config.show.parsed {
            self.show_ast(&ss);
            return Vector::new();
        }
        ss
    }

    pub fn ast_to_hir(&mut self, ss: Vector<ast::Stmt>) -> Vector<hir::Stmt> {
        let ss = ast_to_hir::process(&mut self.ctx1, ss);
        if self.config.show.resolved {
            if !ss.is_empty() {
                self.show_hir(&ss);
            }
            return Vector::new();
        } else {
            ss
        }
    }

    pub fn infer(&mut self, ss: Vector<hir::Stmt>) -> Vector<hir::Stmt> {
        let ss = hir_type_inference::process(&mut self.ctx3, ss);
        if self.config.show.inferred {
            if !ss.is_empty() {
                self.show_hir(&ss);
            }
            return Vector::new();
        } else {
            ss
        }
    }

    pub fn patcomp(&mut self, ss: Vector<hir::Stmt>) -> Vector<hir::Stmt> {
        let ss = hir_patcomp::process(&mut self.ctx4, ss);
        if self.config.show.patcomped {
            if !ss.is_empty() {
                self.show_hir(&ss);
            }
            return Vector::new();
        } else {
            ss
        }
    }

    pub fn monomorphise(&mut self, ss: Vector<hir::Stmt>) -> Vector<hir::Stmt> {
        let ss = hir_monomorphise::process(&mut self.ctx5, ss);
        if self.config.show.monomorphised {
            if !ss.is_empty() {
                self.show_hir(&ss);
            }
            return Vector::new();
        } else {
            ss
        }
    }

    pub fn interpret(&mut self, ss: Vector<hir::Stmt>) {
        hir_interpreter::process(&mut self.ctx6, ss)
    }

    pub fn clear_caches(&mut self) {
        self.ctx6.ctx10.clear_caches();
    }

    pub fn show_caches(&mut self) {
        self.ctx6.ctx10.show_caches();
    }

    pub fn stmts(&self) -> Vector<Stmt> {
        self.ctx5.stmts.clone()
    }

    pub fn compile_prelude(&mut self) -> Result<()> {
        let stmts = ast_prelude::prelude();
        let stmts = self.ast_to_hir(stmts);
        let stmts = self.infer(stmts);
        let stmts = self.patcomp(stmts);
        let stmts = self.monomorphise(stmts);
        if self.has_errors() {
            self.emit_errors();
            panic!("Prelude should have compiled successfully.");
        }
        if self.config.show.prelude {
            codegen::Context::stderr()
                .debug()
                .writeln(&stmts, write_hir::write)?;
        }
        self.interpret(stmts);
        Ok(())
    }

    pub fn has_errors(&self) -> bool {
        self.ctx0.diagnostics.has_errors()
            || self.ctx1.diagnostics.has_errors()
            || self.ctx2.diagnostics.has_errors()
            || self.ctx3.diagnostics.has_errors()
            || self.ctx4.diagnostics.has_errors()
            || self.ctx5.diagnostics.has_errors()
            || self.ctx6.ctx8.diagnostics.has_errors()
    }

    pub fn emit_errors(&mut self) {
        let sources = &mut self.ctx0.sources;
        let opt = &self.config;
        self.ctx0.diagnostics.emit_errors(sources, opt);
        self.ctx1.diagnostics.emit_errors(sources, opt);
        self.ctx2.diagnostics.emit_errors(sources, opt);
        self.ctx3.diagnostics.emit_errors(sources, opt);
        self.ctx4.diagnostics.emit_errors(sources, opt);
        self.ctx5.diagnostics.emit_errors(sources, opt);
        self.ctx6.ctx7.diagnostics.emit_errors(sources, opt);
        self.ctx6.ctx8.diagnostics.emit_errors(sources, opt);
        self.ctx6.ctx9.diagnostics.emit_errors(sources, opt);
        self.ctx6.ctx10.diagnostics.emit_errors(sources, opt);
    }

    pub fn clear(&mut self) {
        self.ctx0 = Default::default();
        self.ctx1 = Default::default();
        self.ctx2 = Default::default();
        self.ctx3 = Default::default();
        self.ctx4 = Default::default();
        self.ctx5 = Default::default();
        self.ctx6 = Default::default();
        self.ctx6.ctx7 = Default::default();
        self.ctx6.ctx8 = Default::default();
        self.ctx6.ctx9 = Default::default();
        self.ctx6.ctx10 = Default::default();
    }

    pub fn print_value_type(&self, vt: &(Value, Type)) -> Result<(), std::io::Error> {
        codegen::Context::stderr()
            .colors(true)
            .writeln(vt, write_value::write_value_type)?;
        Ok(())
    }

    pub fn status(&self) {}

    pub fn show_ast(&self, ast: &Vector<ast::Stmt>) -> Result<()> {
        codegen::Context::stderr()
            .with_opt(self.config.show)
            .writeln(ast, write_ast::write)?;
        Ok(())
    }

    pub fn show_hir(&self, hir: &Vector<hir::Stmt>) -> Result<()> {
        codegen::Context::stderr()
            .with_opt(self.config.show)
            .writeln(hir, write_hir::write)?;
        Ok(())
    }

    pub fn show_mlir(&self, mlir: &Vector<mlir::Item>) -> Result<()> {
        codegen::Context::stderr()
            .with_opt(self.config.show)
            .writeln(mlir, write_mlir::write)?;
        Ok(())
    }

    pub fn show_rust(&self, rust: &Vector<rust::Item>) -> Result<()> {
        codegen::Context::stderr()
            .with_opt(self.config.show)
            .writeln(rust, write_rust::write)?;
        Ok(())
    }
}
