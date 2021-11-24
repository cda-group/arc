# Functions

Functions are written with the `def` keyword.

```grammar
Def ::= "def" [Name] [Generics]? [Params] ":" [Type] [Body]
```

## Examples

### Functional functions

```
{{#include ../../../arc-lang/examples/fib-functional.arc:example}}
```

### Imperative functions

```
{{#include ../../../arc-lang/examples/fib-imperative.arc:example}}
```

### Declare-after-use

```
{{#include ../../../arc-lang/examples/even-odd.arc:example}}
```
