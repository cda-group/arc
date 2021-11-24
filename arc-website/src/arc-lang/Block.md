# Blocks

A **block** is a sequence of statements optionally terminated by an expression which the block evaluates into. If no expression is specified, then the block evaluates into unit.

```grammar
Block ::= "{" [Stmt]* [Expr]? "}"
```

## Examples

```arc-lang
{{#include ../../../arc-lang/examples/blocks.arc:example}}
```
