# Assignments

Variables can be assigned as mutable with `var` and immutable with `val`.

```grammar
Assign ::=
 | "val" [Pattern] (":" [Type])? "=" [Expr] ";"
 | "var" [Pattern] (":" [Type])? "=" [Expr] ";"
```

## Examples

```arc-lang
{{#include ../../../arc-lang/examples/assign.arc:example}}
```
