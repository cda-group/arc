# Timers

A **timer** is an internal port which can be *scheduled* to receive an event *after* a specified duration.

```text
Value ::=
  | Value 'after' Value
  | ..

Type ::=
  | Type 'after' Type
  | ..

Pattern ::=
  | Pattern 'after' Pattern
  | ..

Expr ::=
  | Expr 'after' Expr  # Associate an expression with a duration
  | 'trigger' Expr     # Schedule timer
  | ..

TaskItem ::=
  | 'port' Name ( '(' Type ')' )? ';'  # Internal port
```

## Semantics

Timed values, types, patterns, and expressions, are all desugared as follows:

```text
------------------------------------(DesugarAfter)
Γ ⊢ t0 after t1  =  {val:t0, dur:t1}
```

## Example

`TummblingWindow` is a task which emits an aggregate over a stream every `dur` for the last `dur`.

```text
task TumblingWindow[A, T](init: A, dur: duration, f: fun(A, T): A): (In(~T)) -> (Out(~A)) {
    var agg = init;
    port Tumble;  # A port for triggering window-aggregation results
    trigger Tumble after dur;
    on {
        In(event) => agg = f(agg, event),
        Tumble => {
            emit Out(agg);
            agg = init;
            trigger Tumble after dur
        }
    }
}
```
