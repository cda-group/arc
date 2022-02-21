# Basics

Functions in Arc-Lang are written using the `def` keyword.

```arc-lang
{{#include ../../../arc-lang/examples/fib-functional.arc:example}}
```

Immutable variables can be defined with the `val` keyword and mutable ones with the `var` keyword.

```arc-lang
{{#include ../../../arc-lang/examples/fib-imperative.arc:example}}
```

Common values are supported, including interpolated strings, percentage, duration, and datetime literals.

```arc-lang
{{#include ../../../arc-lang/examples/values.arc:example}}
```

The declaration order of items is insignificant. A function can call another function defined below it in the source code.

```arc-lang
{{#include ../../../arc-lang/examples/even-odd.arc:example}}
```

Enums (also known as disjoint or discriminated unions) are supported:

```arc-lang
{{#include ../../../arc-lang/stdlib/stdlib.arc:option}}
```

Three forms of polymorphism are supported. First, parametric polymorphism allows functions to behave equivalently for different types of parameters:

```arc-lang
{{#include ../../../arc-lang/examples/identity-function.arc:example}}
```

Second, overloading allows functions to behave differently for different types of parameters:

```arc-lang
{{#include ../../../arc-lang/examples/overload-plus.arc:example}}
```

Overloading is achieved through the use of type classes:

```arc-lang
{{#include ../../../arc-lang/examples/monoid.arc:example}}
```

Finally, row polymorphism allows subtyping of records:

```arc-lang
{{#include ../../../arc-lang/examples/row-polymorphism-shape.arc:example}}
```

While the language aims to be general, it also has its limits. Therefore it is also possible to import externally defined types and functions.

```arc-lang
{{#include ../../../arc-lang/stdlib/stdlib.arc:string}}
```
