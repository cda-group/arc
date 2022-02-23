# Body

The body of a function or task can be written either inline using `=` syntax or directly as a block.

```
Body ::=
  | "=" [Expr] ";"
  | [Block]
```

## Examples

```arc-lang
{{#include ../../../arc-lang/examples/body.arc:example}}
```
