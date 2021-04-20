# Values

A **value** is the result of evaluating an **expression**.

```text
Value ::=
  | Literal                                         # Literal-value
  | '{' (Name ':' Value ',')+ '}'                   # Record-value
  | '(' (Value ',')+ ')'                            # Tuple-value
  | 'fun' '(' (Pattern ':' Type ',')+ ')' ':' Expr  # Lambda-value
  | Value? '..' ('='? Value)?                       # Range-value
  | Path                                            # Item-value
  | Path '(' Value ')'                              # Enum-value
```

## Examples

Some examples of different values:

```
{x:5, y:"foo"} # Record
(5, "foo")     # Tuple
fun(x): x+1    # Lambda
0..100         # Exclusive Range
0..=100        # Inclusive Range
```
