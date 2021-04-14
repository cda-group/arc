# Expressions

An **expression** is syntactic construct which can be evaluated into a **value**. Evaluation is always *eager*.

```text
Expr ::=
  | Name                                                       # Variable reference
  | Path                                                       # Item reference
  | Literal                                                    # Literal
  | '{' (Name ':' Expr ',' )+ '}'                              # Record-construction
  | '(' (Expr',')+ ')'                                         # Tuple-construction
  | Expr? '..' ('='? Expr)?                                    # Range-construction
  | Expr BinOp Expr                                            # Binary operation
  | 'var' Pattern (':' Type)? '=' Expr                         # Mutable-binding
  | 'val' Pattern (':' Type)? '=' Expr                         # Immutable-binding
  | 'if' Expr '{' Expr '}' ('else' '{' Expr '}')?              # If-else-expression
  | 'match' Expr '{' (Pattern ('if' Expr)? '=>' Expr ',')+ '}' # Match-expression
  | 'for' Pattern 'in' Expr '{' Expr '}'                       # For-loop
  | 'while' Expr '{' Expr '}'                                  # While-loop
  | 'loop' '{' Expr '}'                                        # Infinite loop
  | 'break' | 'continue' | 'return'                            # Control flow
  | 'try' Expr 'catch' Pattern '=>' Expr ('finally' Expr)?     # Exceptions
  | '[' Expr 'for' Pattern 'in' Expr ('if' Expr)* ']'          # Comprehension
  | 'fun' '(' (Pattern ':' Type ',')+ ')' ':' Expr             # Lambda construction
  | '-' Expr                                                   # Negation
  | 'not' Expr                                                 # Logical
  | Expr '(' (Expr ',')* ')'                                   # Function call
  | Path '(' Expr ')'                                          # Enum-variant construction
  | Expr '.' Name                                              # Field projection
  | Expr '.' [0-9]+                                            # Index projection
  | Expr 'as' Type                                             # Type cast

```

The binary operators are as follows:

```text
BinOp ::=
  | '+'   | '-'  | '*'   | '/'    | '**'  | '%'     # Arithmetic 
  | '=='  | '!=' | '<'   | '>'    | '<='  | '>='    # Equality and Relational
  | 'and' | 'or' | 'xor' | 'band' | 'bor' | 'bxor'  # Logical and Bitwise
  | ';'                                             # Sequential composition
```
