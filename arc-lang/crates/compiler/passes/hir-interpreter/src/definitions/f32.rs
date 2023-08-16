pub fn define(builder: &mut super::Bifs) {
    builder
        .f("add_f32", |ctx, t, v| {
            let v0 = v[0].as_f32();
            let v1 = v[1].as_f32();
            (v0 + v1).into()
        })
        .f("sub_f32", |ctx, t, v| {
            let v0 = v[0].as_f32();
            let v1 = v[1].as_f32();
            (v0 - v1).into()
        })
        .f("mul_f32", |ctx, t, v| {
            let v0 = v[0].as_f32();
            let v1 = v[1].as_f32();
            (v0 * v1).into()
        })
        .f("div_f32", |ctx, t, v| {
            let v0 = v[0].as_f32();
            let v1 = v[1].as_f32();
            (v0 / v1).into()
        });
}
