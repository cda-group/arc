# Extern Functions

An **extern function** is a function-declaration whose implementation is defined externally, outside of Arc-Lang, inside Rust.

```grammar
ExternDef ::= "extern" "def" [Name] [Generics]? "(" [Type]","* ")" ":" [Type] ";"
```

## Examples

```arc-lang
{{#include ../../../arc-lang/stdlib/stdlib.arc:string}}
```
