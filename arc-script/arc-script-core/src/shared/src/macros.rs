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
        $expr:expr , $head:ident$(::$tail:ident)*(_)
    } => {
        if let $head$(::$tail)*(x) = $expr {
            Some(x)
        } else {
            None
        }
    }
}
