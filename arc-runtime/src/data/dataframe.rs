use crate::prelude::*;

pub mod Data {
    use crate::prelude::*;

    #[derive(Clone, From, Deref, DerefMut, Debug, Send, Sync, Unpin, Trace)]
    #[from(forward)]
    pub struct DataFrame(pub Gc<ConcreteDataFrame>);

    #[derive(Debug, NoTrace, Send, Sync, Unpin)]
    pub struct ConcreteDataFrame(pub polars::frame::DataFrame);
}

mod sendable {
    use crate::prelude::*;

    #[derive(Clone, From, Send, Serialize, Deserialize)]
    #[from(forward)]
    pub struct String(pub ConcreteString);

    pub type ConcreteString = Box<str>;
}

impl Data for Data::DataFrame {
    type T = ();
    fn into_sendable(&self, ctx: Context<impl Execute>) -> Self::T {
        panic!("DataFrame is not sendable")
    }
}

pub use Data::DataFrame;

impl DataFrame {
    pub fn new(ctx: Context<impl Execute>) -> Self {
        ctx.heap()
            .allocate(Data::ConcreteDataFrame(
                polars::frame::DataFrame::new::<polars::series::Series>(vec![]).unwrap(),
            ))
            .into()
    }
}
