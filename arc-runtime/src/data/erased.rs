use crate::prelude::*;

#[derive(Unpin, Sync, Send, Clone, Copy, Trace, Serialize, Deserialize)]
pub struct Erased(pub Gc<ConcreteErased>);

#[derive(Unpin, Sync, Send, Trace, Serialize, Deserialize)]
pub struct ConcreteErased(#[serde(with = "serde_traitobject")] pub Box<dyn DynData>);

impl Debug for Erased {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("<erased>").finish()
    }
}

/// NOTE: The object inside must be convertible to `sendable::Erased`.
impl DynData for Erased {
    fn copy_to(self, src: Heap, dst: Heap) -> Self {
        todo!()
        //         (self.0).0.into_sendable(ctx)
    }
}

impl Erased {
    pub fn erase<T: DynData>(x: T, ctx: Context<impl Execute>) -> Self {
        Erased(
            ctx.heap()
                .allocate(ConcreteErased(Box::new(x) as Box<dyn DynData>)),
        )
    }

    pub fn unerase<T: Data>(self, ctx: Context<impl Execute>) -> T {
        let raw = Box::into_raw((self.0).0.clone());
        let raw = raw as *const dyn DynData as *const T as *mut T;
        unsafe { *Box::from_raw(raw) }
    }
}
