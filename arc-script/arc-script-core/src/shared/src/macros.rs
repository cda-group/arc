/// Extracts a pattern from a value forcefully.
#[macro_export]
macro_rules! get {
    {
        $expr:expr , $head:ident$(::$tail:ident)*($($tt:tt),*)
    } => {
        if let $head$(::$tail)*($($tt),*) = $expr {
            ($($tt),*)
        } else {
            unreachable!()
        }
    }
}

/// Extracts a pattern from a value, returns `Some(_)` if it exists, else `None`.
#[macro_export]
macro_rules! map {
    {
        $expr:expr , $head:ident$(::$tail:ident)*
    } => {
        if let $head$(::$tail)*(x) = $expr {
            Some(x)
        } else {
            None
        }
    }
}

/// A wrapper around panic!() which captures location info.
#[macro_export]
macro_rules! ice {
    { } => {
        panic!("{}:{}:{} internal compiler error", file!(), line!(), column!())
    };
    { $($arg:tt)+ } => {
        panic!("{}:{}:{} internal compiler error: {}", file!(), line!(), column!(), format_args!($($arg)+))
    };
}

/// A wrapper around todo!() which captures location info.
#[macro_export]
macro_rules! todo {
    { } => {
        panic!("{}:{}:{} not yet implemented", file!(), line!(), column!())
    };
    { $($arg:tt)+ } => {
        panic!("{}:{}:{} not yet implemented: {}", file!(), line!(), column!(), format_args!($($arg)+))
    };
}

/// Macro for implementing lowerings.
#[macro_export]
macro_rules! lower {
    {
        [$self:ident, $ctx:ident, $repr:ident]
        $($from:ty => $into:ty { $($tt:tt)+ } ,)*
    } => {
        $(
            impl Lower<$into, Context<'_>> for $from {
                fn lower(&self, $ctx: &mut Context<'_>) -> $into {
                    let $self = self;
                    tracing::trace!("{:<14} => {:<16}: {}", stringify!($from), stringify!($into), $ctx.$repr.pretty($self, $ctx.info));
                    $($tt)+ 
                }
            }
        )*
    }
}
