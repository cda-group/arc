use crate::prelude::*;

pub mod sharable {
    use crate::prelude::*;

    #[derive(
        Clone, From, Deref, DerefMut, Debug, Collectable, Finalize, Send, Sync, Unpin, NoTrace,
    )]
    #[from(forward)]
    pub struct DataFrame(pub Gc<ConcreteDataFrame>);

    #[derive(Debug, Collectable, NoTrace, Finalize, Send, Sync, Unpin)]
    pub struct ConcreteDataFrame(pub polars::frame::DataFrame);

    impl Alloc<DataFrame> for ConcreteDataFrame {
        fn alloc(self, ctx: Context) -> DataFrame {
            DataFrame(ctx.mutator().allocate(self, AllocationSpace::New).into())
        }
    }
}

mod sendable {
    use crate::prelude::*;

    #[derive(Clone, From, Send, Serialize, Deserialize)]
    #[from(forward)]
    pub struct String(pub ConcreteString);

    pub type ConcreteString = Box<str>;
}

impl DynSharable for sharable::DataFrame {
    type T = ();
    fn into_sendable(&self, ctx: Context) -> Self::T {
        panic!("DataFrame is not sendable")
    }
}

pub use sharable::DataFrame;

impl DataFrame {
    pub fn new(ctx: Context) -> Self {
        sharable::ConcreteDataFrame(
            polars::frame::DataFrame::new::<polars::series::Series>(vec![]).unwrap(),
        )
        .alloc(ctx)
    }
}
