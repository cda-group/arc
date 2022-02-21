# Type Class Instances

Type class instances are written with the `instance` keyword.

```grammar
TypeClass ::= "class" [Name] [Generics]? "{" [Def]","+ "}"
```

## Examples

```arc-lang
{{#include ../../../arc-lang/examples/type-class.arc:instance}}
```
