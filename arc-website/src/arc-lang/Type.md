# Types

All expressions in arc-lang have a statically inferred type which indicates what set of values they evaluate into. Types of items and variables can be inferred, and thus do not need to be annotated unless desired.

```grammar
Type ::=
  | "#{" ([Name] ":" [Type])","+ "}"     # Record-type
  | "(" [Type]","+ ")"             # Tuple-type
  | "fun" "(" [Type]","+ ")" ":" [Type]  # Function-type
  | [Type]? ".." ("="? [Type])?       # Range-type
  | [Path] ("[" [Type]","* "]")?     # Item-type (with optional type parameters)
```

## Examples

Some examples of different types:

```arc-lang
{{#include ../../../arc-lang/examples/types.arc:record}}
{{#include ../../../arc-lang/examples/types.arc:tuple}}
{{#include ../../../arc-lang/examples/types.arc:function}}
{{#include ../../../arc-lang/examples/types.arc:inclusive_range}}
{{#include ../../../arc-lang/examples/types.arc:exclusive_range}}
```

# Standard types

The following types are provided in the prelude of Arc-Lang:

* `i8`, `i16`, `i32`, `i64` (Machine integers)
* `u8`, `u16`, `u32`, `u64` (Machine unsigned integers)
* `f16`, `bf16`, `f32`, `f64` (Machine floating points)
* `bignum` (Arbitrary sized integer)
* `bool` (Booleans)
* `unit` (Unit)
* `char`, `str` (UTF8-encoded chars and strings)
* `Stream[T]`, `KStream[K, T]` (Data streams)
