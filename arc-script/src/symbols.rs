use crate::ast::*;
use crate::error::CompilerError;
use lasso::Rodeo;
use DeclKind::*;

/// The symbol-table is used to store symbolic identifiers for functions, let-expressions,
/// and type-aliases outside of the AST. Each symbols in the symbol table additionally stores a
/// type.
///
/// The symbol-table is constructed during parsing (i.e., while building the AST). To this end,
/// parsing of functions, let-expressions, and type-aliases must be separated into a declaration
/// and definition step. For example (see `grammar.lalrpop`):
/// ```ignore
/// LetDecl: (Ident, Box<Expr>) = <VarSymbol> <OptionalType> "=" <Box<Arg>> => {
///   let (sym, ty, arg) = (<>);
///   stack.push_scope();
///   let id = table.insert(Decl::new(sym, ty, VarDecl));
///   stack.bind_local(sym, id);
///   (id, arg)
/// };
///
/// LetDef: ExprKind = <LetDecl> <Box<Body>> => {
///   let ((id, arg), body) = (<>);
///   stack.pop_scope();
///   Let(id, arg, body)
/// };
///
/// Var: ExprKind = <VarSymbol> => {
///   stack.lookup(<>)
///        .map(Var)
///        .unwrap_or(ExprErr)
/// };
/// ```
/// When parsing a declaration, any symbol encountered is inserted into the symbol-table. By
/// inserting, a unique identifier is returned as a key to the symbol. The idea is to only store
/// the unique identifiers in the AST, and not the actual symbols, for space-efficiency. The
/// Decl/Def distinction ensures that the function's name and parameters can be inserted into the
/// symbol table before they are used (in the body).
///
/// The symbol is also inserted, along with its key, into a scope in the symbol-stack. The
/// symbol-stack stores each scope and its bindings which allows resolving the symbol of any
/// encountered variable/nominal-type/function-call to its closest binding. One additional note is
/// that functions and type-definitions are, like in Rust, bound in a global namespace and variables
/// from let-expressions in a local namespace. When binding into the local namespace, symbols are
/// inserted into the innermost (last) scope of the stack. When binding into the global namespace,
/// symbols are inserted into the outermost (first) scope of the stack. If a symbol with the same
/// name already exists in that scope, a name-clash error is emitted.
///
/// A new scope is pushed onto the symbol-stack when parsing the declaration of a function or
/// let-expression, and popped when parsing its definition. This means shadowing is allowed between
/// variables.

/// Maps symbolic identifiers to their declarations
#[derive(Default)]
pub struct SymbolTable {
    /// Maps Ident(ifiers) to declarations
    pub decls: Vec<Decl>,
    /// Maps SymbolKeys to SymbolNames, i.e., stores strings outside of the AST.
    pub intern: Rodeo,
}

impl SymbolTable {
<<<<<<< HEAD
    pub fn new() -> Self { Self { decls: Vec::new() } }
=======
    pub fn new() -> Self {
        Self {
            decls: Vec::new(),
            intern: Rodeo::new(),
        }
    }
>>>>>>> a2b70c8... Add interner for uniquing strings

    pub fn insert(&mut self, decl: Decl) -> Ident {
        let id = Ident(self.decls.len());
        self.decls.push(decl);
        id
    }

    pub fn intern(&mut self, symbol: SymbolName) -> SymbolKey {
        self.intern.get_or_intern(symbol)
    }

    pub fn resolve(&self, symbol: &SymbolKey) -> SymbolName {
        self.intern.resolve(symbol)
    }

    pub fn genvar(&mut self, tv: TypeVar) -> Ident {
        let id = Ident(self.decls.len());
        let key = self.intern.get_or_intern(&format!("x{}", id.0));
        self.decls.push(Decl::new(key, tv, VarDecl));
        id
    }

<<<<<<< HEAD
    pub fn get_mut(&mut self, ident: &Ident) -> &mut Decl { self.decls.get_mut(ident.0).unwrap() }

    pub fn get(&self, ident: &Ident) -> &Decl { self.decls.get(ident.0).unwrap() }
=======
    pub fn get_decl_mut(&mut self, ident: &Ident) -> &mut Decl {
        self.decls.get_mut(ident.0).unwrap()
    }

    pub fn get_decl_name(&self, ident: &Ident) -> &str {
        self.resolve(&self.get_decl(&ident).sym)
    }

    pub fn get_decl(&self, ident: &Ident) -> &Decl {
        self.decls.get(ident.0).unwrap()
    }
>>>>>>> a2b70c8... Add interner for uniquing strings

    pub fn debug(&self) {
        for (i, decl) in self.decls.iter().enumerate() {
            let name = self.intern.resolve(&decl.sym);
            match decl.kind {
                VarDecl => println!("[var]  {} => {}", i, name),
                FunDecl => println!("[fun]  {} => {}", i, name),
                TypeDecl => println!("[type] {} => {}", i, name),
                TaskDecl(_) => println!("[task] {} => {}", i, name),
            }
        }
    }
}

pub type Scope = Vec<(SymbolKey, Ident)>;
pub struct SymbolStack {
    scopes: Vec<Scope>,
}

impl SymbolStack {
    pub fn new() -> Self {
        let mut scopes = Vec::new();
        let globals = Vec::new();
        scopes.push(globals);
        Self { scopes }
    }

    pub fn lookup(&self, needle: SymbolKey) -> Option<Ident> {
        self.scopes.iter().rev().find_map(|scope| {
            scope
                .iter()
                .rev()
                .find_map(|&(name, ident)| if name == needle { Some(ident) } else { None })
        })
    }

    pub fn bind_local(&mut self, name: SymbolKey, id: Ident) {
        self.locals().push((name, id))
    }

    pub fn bind_global(&mut self, name: SymbolKey, ident: Ident) -> Result<(), CompilerError> {
        if self.globals().iter().any(|(needle, _)| name == *needle) {
            Err(CompilerError::NameClash)
        } else {
            self.globals().push((name, ident));
            Ok(())
        }
    }

    pub fn globals(&mut self) -> &mut Scope {
        self.scopes.first_mut().unwrap()
    }

    pub fn locals(&mut self) -> &mut Scope {
        self.scopes.last_mut().unwrap()
    }

    pub fn push_scope(&mut self) { self.scopes.push(Vec::new()) }

    pub fn pop_scope(&mut self) { self.scopes.pop(); }
}
