# Types

A **type** is a syntactic construct which represents a set of **values** with common behavior. All expressions in Arc-Script have a statically inferred type which indicates what set of values they evaluate into. Items types such as functions need to be annotated with types in their signature, but can be generic.

```text
Type ::=
  | Scalar                              # Scalar-type
  | '{' '(' Name ':' Type ',')+ '}'     # Record-type
  | '(' (Type ',')+ ')'                 # Tuple-type
  | 'fun' '(' (Type ',')+ ')' ':' Type  # Function-type
  | Type? '..' ('='? Type)?             # Range-type
  | Path ('[' (Type ',')* ']')?         # Item-type (with optional type parameters)
```

## Examples

Some examples of different types:

```text
{x:i32, y:str}  # Record-type
(i32, str)      # Tuple-type
fun(x:i32): i32 # Lambda-type
i32..i32        # Exclusive Range-type
i32..=i32       # Inclusive Range-type
```
