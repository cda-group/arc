# Timeouts

A **timeout** is a control-event which is triggered after no stream-event is received within a specified duration. Internally, a counter within the task is ticking down. When it reaches zero, the `after` (timeout) clause is fired. The counter is set to the specified duration whenever an event is received. The counter starts ticking as soon as it receives its first event.

```text
TaskItem ::=
  | 'on' '{' (Pattern ('if' Expr)? '=>' Expr ',')+ (Control ',')* '}'  # Event handler
  | ..

Control ::=
  | 'after' Expr '=>' Expr  # Timeout
  | ..
```

## Examples

Timeouts allow the definition of *session windows*. For example:

```text
# A window which outputs an aggregate (and resets) after a timeout of not receiving events.
task SessionWindow[A,T](init: A, dur: duration, binop: fun(A, T): A) ~T -> ~A {
    var agg = init;
    on {
        event => agg = binop(agg, event),
        after dur => {
            emit agg;
            agg = init
        }
    }
}
```
