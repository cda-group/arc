# Extern types

An **extern type** is an abstract data type with methods whose implementations are defined in Rust.

```text
Item ::=
  | 'extern' 'type' Name '(' (Name ':' Type ',')* ')' '{' # Abstract Data Type (ADT)
      ('fun' Name '(' (Type ',')* ')' ':' Type ';')*      # Externally implemented method
    } 
  | ...
```

## Example

The following code shows how to define an extern type `DataFrame` in Arc-Script.

```text
# src/my_script.arc

extern type DataFrame() {
    fun sum() -> i32;
}

fun test() {
    val df = DataFrame();
    # ...
    df.sum();
}
```

Here is one way of implementing the type in Rust.

```rust
// src/main.rs

#[arc_script::include("my_script.arc")]
mod my_script {
    #[derive(Clone)]
    pub struct DataFrame {
        concrete: Rc<RefCell<polars::DataFrame>>,
    }

    impl DataFrame {
        pub fn new() -> DataFrame { /* ... */ }
        pub fn sum(self) -> i32 { /* ... */ }
    }
}
```
