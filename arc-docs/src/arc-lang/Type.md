# Types

All expressions in arc-lang have a statically inferred type which indicates what set of values they evaluate into. Types of items and variables can be inferred, and thus do not need to be annotated unless desired.

```grammar
Type ::=
  | "#{" ([Name] ":" [Type])","+ "}"   # Record-type
  | "(" [Type]","+ ")"             # Tuple-type
  | "fun" "(" [Type]","+ ")" ":" [Type]  # Function-type
  | "[" [Type] "]"               # Array-type
  | [Path] ("[" [Type]","* "]")?     # Item-type (with optional type parameters)
```

## Examples

Some examples of different types:

```arc-lang
{{#include ../../../arc-lang/examples/types.arc:record}}
{{#include ../../../arc-lang/examples/types.arc:tuple}}
{{#include ../../../arc-lang/examples/types.arc:array}}
{{#include ../../../arc-lang/examples/types.arc:function}}
```

# Standard types

The following types are provided in the [standard library](https://github.com/cda-group/arc/blob/master/arc-lang/stdlib/stdlib.arc) of Arc-Lang:

```arc-lang
{{#exec grep -F 'extern type' ../arc-lang/stdlib/stdlib.arc}}
```
