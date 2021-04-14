# Extern Functions

An **extern function** is a function-declaration whose implementation is defined externally, outside of Arc-Script, inside Rust.

```text
Item ::=
  | 'extern' 'fun' Name '(' (Name ':' Type ',')* ')' ':' Type ';'  # Extern function declaration
  | ...
```

## Example

The following code declares an extern function `add` for adding two numbers.

```text
# src/my_script.arc

extern fun add(a:i32, b:i32): i32;

fun test(): i32 {
    add(1, 2) # 1+2 = 3
}
```

The implementation is defined inside of Rust, for example:

```rust
// src/main.rs

#[arc_script::include("my_script.arc")]
mod my_script {
    // Anything in this module's namespace is visible to my_script.arc.
    fn add(a:i32, b:i32) -> i32 {
        a + b
    }
}
```
