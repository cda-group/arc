use value::ValueKind::VUnit;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("print", |ctx, t, v| {
            let v0 = v[0].as_string();
            eprintln!("{}", v0.as_ref());
            ().into()
        })
        .f("debug", |ctx, t, v| {
            let v0 = &v[0];
            eprintln!("{:?}", v0);
            ().into()
        })
        .f("dataflow", |ctx, t, v| {
            todo!();
            ().into()
        })
        .f("connect", |ctx, t, v| {
            let v0 = v[0].as_string();
            match kafka::context::Context::new(Some(v0.as_ref().to_string())) {
                Ok(v) => {
                    eprintln!("Connected to Kafka broker {}", v0);
                    ctx.ctx11 = Some(v);
                }
                Err(v) => eprintln!("{}", v),
            }
            ().into()
        })
        .f("topics", |ctx, t, v| {
            if let Some(ctx) = ctx.ctx11.as_mut() {
                if let Err(e) = ctx.list() {
                    eprintln!("{}", e);
                }
            } else {
                eprintln!("Kafka not connected");
            }
            ().into()
        })
        .f("bifs", |ctx, t, v| {
            write_ast::write(
                &mut codegen::Context::stderr().colors(true),
                &ast_prelude::prelude(),
            );
            ().into()
        })
        .f("typeof", |ctx, t, v| {
            let t0 = &t[0];
            let ctx = codegen::Context::stderr()
                .colors(true)
                .writeln(t0, write_hir::write_type)
                .unwrap();
            ().into()
        });
}
