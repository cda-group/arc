# Uses

A **use**-item imports a name into the current namespace and optionally aliases it.


```grammar
Use ::= "use" [Path] ("as" [Name])?;
```

## Examples

The following code creates a `Person`-type and an alias and use it as a `Human`.

```text
{{#include ../../../arc-lang/examples/uses.arc:example}}
```
