use crate::prelude::*;
use codespan::Span;
use lasso::Rodeo;

/// The symbol-table is used to store symbolic identifiers for functions, let-expressions,
/// and type-aliases outside of the AST. Each symbols in the symbol table additionally stores a
/// type.
///
/// The symbol-table is constructed during parsing (i.e., while building the AST). To this end,
/// parsing of functions, let-expressions, and type-aliases must be separated into a declaration
/// and definition step. For example (see `grammar.lalrpop`):
/// ```text
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
    /// Maps identifiers (indices) to declarations.
    pub decls: Vec<Decl>,
    /// Maps symbols to symbol buffers, i.e., stores strings outside of the AST.
    pub intern: Rodeo,
}

impl SymbolTable {
    /// Creates a new Symbol Table
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a new declaration into the symbol table and returns its identifier
    pub fn insert(&mut self, decl: Decl) -> Ident {
        let id = Ident(self.decls.len());
        self.decls.push(decl);
        id
    }

    /// Interns a symbol buffer and returns its symbol.
    pub fn intern(&mut self, Spanned(l, buf, r): Spanned<SymbolBuf>) -> Symbol {
        let span = Span::new(l as u32, r as u32);
        let key = self.intern.get_or_intern(buf);
        Symbol { key, span }
    }

    /// Resolves a symbol into a symbol buffer.
    pub fn resolve(&self, symbol: &Symbol) -> SymbolBuf {
        self.intern.resolve(&symbol.key)
    }

    /// Generates a new variable with type `tv`.
    /// The span of the symbol is unknown (for now).
    pub fn genvar(&mut self, tv: TypeVar) -> Ident {
        let id = Ident(self.decls.len());
        let key = self.intern.get_or_intern(&format!("x{}", id.0));
        let span = Span::new(0, 0);
        let sym = Symbol::new(key, span);
        self.decls.push(Decl::new(sym, tv, VarDecl));
        id
    }

    /// Retrieves the mutable declaration of an identifier.
    pub fn get_decl_mut(&mut self, ident: &Ident) -> &mut Decl {
        self.decls.get_mut(ident.0).unwrap()
    }

    /// Retrieves the declaration of an identifier.
    pub fn get_decl(&self, ident: &Ident) -> &Decl {
        self.decls.get(ident.0).unwrap()
    }

    /// Retrieves the name of a declaration of an identifier.
    pub fn get_decl_name(&self, ident: &Ident) -> &str {
        self.resolve(&self.get_decl(&ident).sym)
    }

    /// Debug-prints a symbol table.
    #[rustfmt::skip]
    pub fn debug(&self) {
        for (i, decl) in self.decls.iter().enumerate() {
            let name = self.intern.resolve(&decl.sym.key);
            match decl.kind {
                VarDecl     => println!("[var]     {} => {}", i, name),
                VariantDecl => println!("[variant] {} => {}", i, name),
                SourceDecl  => println!("[source]  {} => {}", i, name),
                SinkDecl    => println!("[sink]    {} => {}", i, name),
                FunDecl     => println!("[fun]     {} => {}", i, name),
                TypeDecl    => println!("[type]    {} => {}", i, name),
                TaskDecl    => println!("[task]    {} => {}", i, name),
            }
        }
    }
}

pub type Scope = Vec<(Symbol, Ident)>;

pub struct SymbolStack {
    scopes: Vec<Scope>,
    /// The innermost scope where item-identifiers are stored. This is needed for definitions
    /// which contain definitions, for example operators.
    item_scopes: usize,
}

impl Default for SymbolStack {
    fn default() -> Self {
        let mut scopes = Vec::new();
        let globals = Vec::new();
        scopes.push(globals);
        let item_scopes = scopes.len();
        Self {
            scopes,
            item_scopes,
        }
    }
}

impl SymbolStack {
    /// Returns a new symbol stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the identifier of a symbol.
    pub fn lookup(&self, needle: Symbol) -> Option<Ident> {
        self.scopes.iter().rev().find_map(|scope| {
            scope
                .iter()
                .rev()
                .find_map(|&(name, ident)| if name == needle { Some(ident) } else { None })
        })
    }

    pub fn bind_local(&mut self, name: Symbol, id: Ident) {
        self.locals().push((name, id))
    }

    pub fn bind_item(&mut self, name: Symbol, id: Ident) -> Result<(), CompilerError> {
        if self
            .items()
            .any(|scope| scope.iter().any(|(needle, _)| name == *needle))
        {
            Err(CompilerError::NameClash)
        } else {
            self.items().last().unwrap().push((name, id));
            Ok(())
        }
    }

    pub fn items(&mut self) -> impl Iterator<Item = &mut Scope> {
        self.scopes.iter_mut().take(self.item_scopes)
    }

    pub fn locals(&mut self) -> &mut Scope {
        self.scopes.last_mut().unwrap()
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(Vec::new())
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn push_item_scope(&mut self) {
        self.scopes.push(Vec::new());
        self.item_scopes += 1;
    }

    pub fn pop_item_scope(&mut self) {
        self.scopes.pop();
        self.item_scopes -= 1;
    }
}