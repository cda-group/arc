use super::Value;

use hir::Type;
use serde::Deserialize;
use serde::Serialize;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("zeros", |ctx, t, v| {
            // let v0 = v[0].as_vec().0.into_iter().map(|x| x.as_usize()).collect();
            todo!()
            // Matrix::new(v0)
        })
        .f("insert_axis", |ctx, t, v| {
            todo!()
            // let v0 = v[0].as_matrix();
            // let v1 = v[1].as_usize().0;
            // Matrix(Inner::insert_axis(v0, v1)).into()
        })
        .f("remove_axis", |ctx, t, v| {
            todo!()
            // let v0 = v[0].as_matrix();
            // let v1 = v[1].as_usize().0;
            // Matrix(Inner::remove_axis(v0, v1)).into()
        })
        .f("into_vec", |ctx, t, v| {
            // let v0 = v[0].as_matrix().0;
            // let v0 = v0.into_vec();
            // let v0 = v0.iter().map(|x| Value::new(x)).collect();
            // v0.into()
            todo!()
        });
}
