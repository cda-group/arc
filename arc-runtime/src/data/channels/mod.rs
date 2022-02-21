use crate::context::Context;
use kompact::prelude::KompactSystem;

// pub mod remote {
//     pub mod multicast;
//     pub mod parallel;
//     pub mod window;
// }
pub mod local {
    pub mod multicast;
    pub mod parallel;
    //     pub mod window;
}

/// A trait for a channel which is implemented for both endpoints (`Pushable` and `Pullable`).
pub trait Channel {
    type Pushable;
    type Pullable;
    fn channel(ctx: Context) -> (Self::Pushable, Self::Pullable);
}

macro_rules! impl_channel {
    () => {
        impl<T: Sharable> crate::data::channels::Channel for Pushable<T> {
            type Pushable = Self;
            type Pullable = Pullable<T>;

            fn channel(ctx: Context) -> (Self::Pushable, Self::Pullable) {
                channel(ctx)
            }
        }

        impl<T: Sharable> crate::data::channels::Channel for Pullable<T> {
            type Pushable = Pushable<T>;
            type Pullable = Self;

            fn channel(ctx: Context) -> (Self::Pushable, Self::Pullable) {
                channel(ctx)
            }
        }
    };
}

pub(crate) use impl_channel;
