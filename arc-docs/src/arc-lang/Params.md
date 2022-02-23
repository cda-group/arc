# Parameters

Parameters are used in function and task definitions. In general, they do not need to be type annotated.

```grammar
Params ::= "(" [Param]","+ ")"

Param ::= [Pattern] (":" [Type] )?
```

### Examples

```arc-lang
{{#include ../../../arc-lang/examples/params.arc:example}}
```
