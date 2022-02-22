# Type Aliases

A **type alias** is a purely cosmetic alias of a type.

```grammar
TypeAlias ::= "type" [Name] [Generics]? "=" [Type] ";"
```

## Example

The following code defines type aliases for representing lines on a two-dimensional plane, and a function for calculating the length of a line.

```text
{{#include ../../../arc-lang/examples/type-alias.arc:example}}
```
