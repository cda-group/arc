# Type Classes

Type classes are written with the `class` keyword.

```grammar
TypeClass ::= "class" [Name] [Generics]? "{" [[Decl]]","+ "}"

Decl ::= "def" [Name] [Generics]? "(" [Type]","+ ")"
```

## Examples

```arc-lang
{{#include ../../../arc-lang/examples/type-class.arc:class}}
```
