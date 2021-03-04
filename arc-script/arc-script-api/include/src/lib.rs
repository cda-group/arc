#[macro_export]
macro_rules! include {
    {
        $file:literal
    } => {
        ::core::include!(::core::concat!(::core::env!("OUT_DIR"), "/", $file));
    }
}

#[macro_export]
macro_rules! bridge {
    { } => {
        ::core::include!(::core::concat!(::core::env!("OUT_DIR"), "/", file!()));
    }
}
