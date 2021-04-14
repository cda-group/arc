# Functions

A **function** is your ordinary function. It takes some parameters and evaluates its body into a value.

```text
Item ::=
  | 'fun' Name '(' (Name ':' Type ',')* ')' ':' Type '{' Expr '}'  # Function definition
  | ...

Expr ::=
  | Expr '(' (Expr ',')* ')'                                       # Function call syntax
  | ...
```

## Examples

The following code shows how to define the Fibonacci function.

```text
fun fib(n: i32): i32 {
    match n {
        0 => 0,
        1 => 1,
        n => fib(n-2) + fib(n-1)
    }
}

fun test() {
    println(fib(10));
}
```
