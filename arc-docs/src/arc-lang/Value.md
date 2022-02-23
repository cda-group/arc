# Values

A **value** is the result of evaluating an **expression**.

```grammar
Value ::=
  | "#{" ([Name] ":" [Value])","+ "}"              # Record-value
  | "(" [Value]","+ ")"                        # Tuple-value
  | "fun" "(" ([Pattern] ":" [Type])","+ ")" ":" [Expr]  # Lambda-value
  | [Value]? ".." ("="? [Value])?              # Range-value
  | [Path]                               # Item-value
  | [Path] "(" [Value] ")"                     # Enum-value
  | [Literal]

Literal ::=
  | "true" | "false"        # Literal boolean
  | "unit"                # Literal unit
  | "'"[^"'"]"'"              # Literal character
  | """[^"""]*"""             # Literal string
  | ["1"-"9"]["0"-"9"]*         # Literal integer
  | ["1"-"9"]["0"-"9"]*"."["0"-"9"]*  # Literal floating point
  | [[DateTime]]
  | [[Duration]]

DateTime ::=
  | ["0"-"9"]+"-"["0"-"9"]+"-"["0"-"9"]+                                         # Date
  | ["0"-"9"]+"-"["0"-"9"]+"-"["0"-"9"]+"T"["0"-"9"]+":"["0"-"9"]+":"["0"-9]+                    # Date + Time
  | ["0"-"9"]+"-"["0"-"9"]+"-"["0"-"9"]+"T"["0"-"9"]+":"["0"-"9"]+":"["0"-"9"]+("+"|"-")["0"-"9"]+":"["0"-"9"]+  # Date + Time + Zone

Duration ::= 
  | ["0"-"9"]+"ns"  # Nanosecond
  | ["0"-"9"]+"us"  # Microsecond
  | ["0"-"9"]+"ms"  # Millisecond
  | ["0"-"9"]+"s"   # Second
  | ["0"-"9"]+"m"   # Minute
  | ["0"-"9"]+"h"   # Hour
  | ["0"-"9"]+"d"   # Day
  | ["0"-"9"]+"w"   # Week
```

## Examples

Some examples of different values:

```arc-lang
{{#include ../../../arc-lang/examples/values.arc:example}}
```
