# Generics

Items can be parameterised by generic types. Generics can in addition be bounded by type class constraints.

```grammar
Generics ::= "[" [[Generic]]","+ "]"

Generic ::= [Name] [[Bounds]]?

Bounds ::= ":" [Path]"&"+
```

## Examples

### Explicit generic function

```arc-lang
{{#include ../../../arc-lang/examples/generic-function.arc:example}}
```

### Inferred generic function

```arc-lang
{{#include ../../../arc-lang/examples/inferred.arc:example}}
```
