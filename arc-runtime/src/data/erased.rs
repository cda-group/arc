use crate::prelude::*;

mod sharable {
    use crate::prelude::*;

    #[derive(Unpin, Sync, Send, Clone, Collectable, Trace, Finalize)]
    pub struct Erased(pub Gc<ConcreteErased>);

    #[derive(Unpin, Sync, Send, Collectable, Trace, Finalize)]
    pub struct ConcreteErased(pub Box<dyn DynSharable<T = super::sendable::Erased>>);

    impl Debug for Erased {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_tuple("<erased>").finish()
        }
    }

    impl Alloc<Erased> for ConcreteErased {
        fn alloc(self, ctx: Context) -> Erased {
            Erased(ctx.mutator().allocate(self, AllocationSpace::New).into())
        }
    }
}

pub use sharable::Erased;

pub mod sendable {
    use crate::prelude::*;

    #[derive(Clone, From, Serialize, Deserialize)]
    pub struct Erased(pub ConcreteErased);

    #[derive(Clone, Serialize, Deserialize)]
    pub struct ConcreteErased(
        #[serde(with = "serde_traitobject")] pub Box<dyn DynSendable<T = super::sharable::Erased>>,
    );

    impl Debug for Erased {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_tuple("<erased>").finish()
        }
    }
}

/// NOTE: The object inside must be convertible to `sendable::Erased`.
impl DynSharable for sharable::Erased {
    type T = sendable::Erased;
    fn into_sendable(&self, ctx: Context) -> Self::T {
        (self.0).0.into_sendable(ctx)
    }
}

impl DynSendable for sendable::Erased {
    type T = sharable::Erased;
    fn into_sharable(&self, ctx: Context) -> Self::T {
        (self.0).0.into_sharable(ctx)
    }
}

impl sharable::Erased {
    pub fn erase<T: DynSharable<T = sendable::Erased> + 'static>(x: T, ctx: Context) -> Self {
        sharable::ConcreteErased(Box::new(x) as Box<dyn DynSharable<T = sendable::Erased>>)
            .alloc(ctx)
    }

    pub fn unerase<T: Sharable + 'static>(self, ctx: Context) -> T {
        let raw = Box::into_raw((self.0).0.clone());
        let raw = raw as *const dyn DynSharable<T = sendable::Erased> as *const T as *mut T;
        unsafe { *Box::from_raw(raw) }
    }
}

impl sendable::Erased {
    pub fn erase<T: DynSendable<T = sharable::Erased> + 'static>(x: T, ctx: Context) -> Self {
        sendable::ConcreteErased(Box::new(x) as Box<dyn DynSendable<T = sharable::Erased>>).into()
    }

    pub fn unerase<T: Sendable + 'static>(self, ctx: Context) -> T {
        let raw = Box::into_raw((self.0).0);
        let raw = raw as *const dyn DynSendable<T = sendable::Erased> as *const T as *mut T;
        unsafe { *Box::from_raw(raw) }
    }
}
