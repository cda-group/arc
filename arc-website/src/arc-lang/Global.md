# Globals

A **global** is an *immutable* variable which can be referenced by-path.

```grammar
Global ::= "val" [Name] "=" [Value] ";"
```

## Example

The following code initializes two global variables, one by a literal, and another by calling a function.

```arc-lang
{{#include ../../../arc-lang/examples/global.arc:example}}
```
