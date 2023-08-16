#[test]
fn test() {
    let decls = ast_prelude::prelude()
        .into_iter()
        .filter_map(|s| {
            if let ast::StmtKind::SBuiltinDef(_, x, ..) = s.kind {
                Some(x)
            } else {
                None
            }
        })
        .collect::<std::collections::HashSet<String>>();
    let defs = hir_interpreter::definitions::Bifs::new();

    for x in &decls {
        assert!(defs.0.contains_key(x.as_str()), "{} is undefined", x);
    }

    for x in defs.0.keys() {
        assert!(decls.contains(*x), "{} is undeclared", x);
    }
}
