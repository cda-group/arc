# Items

An **item** is a named top-level definition which can be referenced by-path. The ordering of how items are defined in the source code insignificant.

```text
Item ::=
  | 'val' Name '=' Value ';'                                       # Global immutable value
  | 'fun' Name '(' (Name ':' Type ',')* ')' ':' Type '{' Expr '}'  # Function definition
  | 'type' Name '=' Type ';'                                       # Type alias
  | 'extern' 'fun' Name '(' (Name ':' Type ',')* ')' ':' Type ';'  # Extern function declaration
  | 'enum' Name '{' (Name '(' Type ')' ',')* '}'                   # Enumerated type (Sum-type)
  | 'enum' Name '(' Type ')'                                       # Nominal type
  | 'use' Path '(' 'as' Name ')'?;                                 # Import (and optionally alias)
  | ..
```
