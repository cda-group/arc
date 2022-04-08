# Expressions

An **expression** is syntactic construct which can be evaluated into a **value**.

```grammar
Expr ::=
  | [Name]         # Variable reference
  | [Path]         # Item reference
  | [Value]        # Value literal
  | "(" [[BinOp]] ")"    # Binary operator reference
  | "_"            # Placeholder
  | [[Query]]        # Query expression
  | [[Constructor]]  # Constructor expression
  | [[Operation]]    # Operation expression
  | [[ControlFlow]]  # Control flow expression
  | [[DataFlow]]     # Dataflow expression

Constructor ::=
  | "#{" ([Name] ":" [Expr])","* "}"          # Record-construction
  | "(" [Expr]","+ ")"                    # Tuple-construction
  | "[" [Expr]","* "]"                    # Array-construction
  | [Expr]? ".." ("="? [Expr])?           # Range-construction
  | "fun" [Params] (":" [Type])? "=" [Expr]   # Lambda-function construction
  | "task" [Params]? ":" [Params] "=" [Expr]  # Lambda-task construction
  | [Path] "(" [Expr] ")"                 # Enum-variant construction

Operation ::=
  | [Expr] [[BinOp]] [Expr]   # Binary operation
  | [Expr] [[UnOp]]         # Unary operator
  | [Expr] "(" [Expr]","* ")"   # Function call
  | [Expr] "." [Name]       # Field projection
  | [Expr] "." [0-9]+     # Index projection
  | [Expr] "as" [Type]      # Type cast
  | [Expr] "in" [Expr]      # Contains
  | [Expr] "not" "in" [Expr]  # Does not contain

UnOp ::=
  | "-"    # Arithmetic negation
  | "not"  # Logical negation

BinOp ::=
  | "+"   | "-"  | "*"   | "/"    | "**"  | "%"     # Arithmetic
  | "=="  | "!=" | "<"   | ">"    | "<="  | ">="    # Equality and comparison
  | "and" | "or" | "xor" | "band" | "bor" | "bxor"  # Logical and bitwise

ControlFlow ::=
  | "if" [Expr] [Block] ("else" [Block])?                         # If-else-expression
  | "match" [Expr] "{" ([Pattern] ("if" [Expr])? "=>" [Expr])","+ "}"       # Match-expression
  | "for" [Pattern] "in" [Expr] "{" [Expr] "}"                        # For-loop
  | "while" [Expr] [Block]                                    # While-loop
  | "loop" [Block]                                          # Infinite loop
  | "break" | "continue" | "return" [Expr]?                     # Jumps
  | "try" [Expr] "catch" ([Pattern] "=>" [Expr])","+ ("finally" [Expr])?  # Exceptions
  | "[" [Expr] "for" [Pattern] "in" [Expr] ("if" [Expr])* "]"             # Comprehension

DataFlow ::=
  | "receive" [Expr]                        # Selective receive
  | "on" "{" ([Pattern] "in" [Expr] "=>" [Expr])","+ "}"  # Non-selective receive
  | [Expr] "!" [Expr]                         # Emit event

Query ::= "from" ([Pattern] "in" [Expr])","+ [[QueryStmt]]+

QueryStmt ::=
  | "yield" [Expr]                            # Select
  | "where" [Expr]                            # Filter
  | "join" [Expr] ("on" [Expr])?                  # Join
  | "keyby" (([Name] "=")? [Expr])","*              # Partition
  | "compute" ([Name] "=")? [Expr] ("of" [Expr])?     # Aggregation
  | "sort" [Expr] "desc"?                       # Sort
  | "window" [Expr] ("every" [Expr])? ("at" [Expr])?  # Sliding or tumbling window
```

## Operators

Operators are defined as follows, with precedence from highest to lowest:

| Operator                                       | Arity   | Affix   | Associativity | Overloadable? |
| ---------------------------------------------- | -----   | -----   | ------------- | ------------  |
| `return` `break`                               | Unary   | Prefix* |               | No            |
| `fun` `task` `on`                              | Unary   | Prefix  |               | No            |
| `=` `!` `+=` `-=` `%=` `*=` `/=` `**=`         | Binary  | Infix   | None          | No            |
| `in` `not in`                                  | Binary  | Infix   | Left          | No            |
| `..` `..=`                                     | Binary  | Infix   | None          | No            |
| `and` `or` `xor` `bor` `band` `bxor`           | Binary  | Infix   | Left          | Yes           |
| `==` `!=`                                      | Binary  | Infix   | None          | No            |
| `<` `>` `<=` `>=`                              | Binary  | Infix   | None          | No            |
| `-` `+` `%`                                    | Binary  | Infix   | Left          | Yes           |
| `*` `/`                                        | Binary  | Infix   | Left          | Yes           |
| `**`                                           | Binary  | Infix   | Right         | Yes           |
| `not` `-`                                      | Unary   | Prefix  |               | Yes           |
| `as`                                           | Binary  | Infix   | Left          | No            |
| `(exprs)` `[exprs]`                            | Unary   | Postfix |               | No            |
| `.index` `.name` `.name(exprs)` `.name[exprs]` | Unary   | Postfix |               | No            |
| Primary expressions                            | Nullary |         |               | No            |

(*) Operand is optional.

## Builtin Functions

The builtin functions of Arc-Lang are listed here.

```arc-lang
{{#exec grep -F 'extern def' ../arc-lang/stdlib/stdlib.arc}}
```

## Examples

### Basic function calls

```arc-lang
{{#include ../../../arc-lang/examples/basic.arc:example}}
```

### Lambda functions

```arc-lang
{{#include ../../../arc-lang/examples/lambda.arc:example}}
```

### Binary operators

```arc-lang
{{#include ../../../arc-lang/examples/binops.arc:example}}
```

### Comprehensions

```arc-lang
{{#include ../../../arc-lang/examples/comprehensions.arc:example}}
```

### Placeholders

```arc-lang
{{#include ../../../arc-lang/examples/placeholder.arc:example}}
```

### Binary operator lifting

```arc-lang
{{#include ../../../arc-lang/examples/binopref.arc:example}}
```

### String interpolation

```arc-lang
{{#include ../../../arc-lang/examples/interpolate.arc:example}}
```

### Query (with explicit variables)

```arc-lang
{{#include ../../../arc-lang/examples/query.arc:explicit}}
```

### Query (with implicit variables)

```arc-lang
{{#include ../../../arc-lang/examples/query.arc:implicit}}
```
