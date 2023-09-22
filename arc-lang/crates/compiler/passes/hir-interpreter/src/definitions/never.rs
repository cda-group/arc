pub fn define(builder: &mut super::Bifs) {
    builder
        .f("todo", |ctx, t, v| {
            todo!();
        })
        .f("unreachable", |ctx, t, v| {
            unreachable!();
        })
        .f("panic", |ctx, t, v| {
            let v0 = v[0].as_string();
            panic!("{}", v0);
        })
        .f("exit", |ctx, t, v| {
            std::process::exit(0);
        });
}

