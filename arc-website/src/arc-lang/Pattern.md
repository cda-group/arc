# Patterns

A **pattern** is a syntactic construct for deconstructing a **value** and binding its parts to variables.

```grammar
Pattern ::=
  | [Name]                       # Variable binding
  | [Value]                      # Value comparison
  | "{" ([Name] (":" [Pattern])?)","+ "}"  # Record deconstruction
  | "(" [Pattern]","+ ")"              # Tuple deconstruction
  | [Pattern]? ".." ("="? [Pattern])?  # Range deconstruction
  | [Path] "(" [Pattern] ")"           # Variant deconstruction
  | [Pattern] "or" [Pattern]         # Alternation
```

## Examples

### Tuples

```arc-lang
{{#include ../../../arc-lang/examples/tuple-patterns.arc:example}}
```

### Records

```arc-lang
{{#include ../../../arc-lang/examples/record-patterns.arc:example}}
```

### Enums

```arc-lang
{{#include ../../../arc-lang/examples/enum-patterns.arc:example}}
```

### Enums

```arc-lang
{{#include ../../../arc-lang/examples/vector-patterns.arc:example}}
```
