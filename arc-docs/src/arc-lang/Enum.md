# Enums

An **enum** is a nominal type which is a disjoint set (tagged union) of values. Each item of the disjoint set is referred to as a **variant**.

```grammar
Enum ::= "enum" [Name] [Generics]? "{" [[Variant]]","* "}"

Variant ::= [Name] "(" [Type] ")"
```

## Examples

```arc-lang
{{#include ../../../arc-lang/examples/shapes.arc:example}}
```
