# Termination

Tasks are by default *long-running*, meaning that they will continue processing as long as there is input. Tasks can in addition be *short-running*, meaning that they might eventually **terminate** and potentially produce an output value. The output value is a *future* that completes when the task terminates. By terminating, the task also emits an **end-of-stream marker** on all its output channels, indicating that it will produce no more output, and is turned into a **sink**.

```text
Expr ::=
  | 'exit' Expr?  # Terminates the task with an optional return value

Control ::=
  | 'done' '=>' Expr  # Fires when end-of-stream markers are received on all inputs
```

## Example 1

`TakeUntil` is a task which forwards events of a stream until a predicate is satisfied.

```text
task TakeUntil[T](p: fun(T): bool): ~T -> ~T {
    on event => if p(event) {
        exit
    } else {
        emit event
    }
}

# Take until a number above 100 is encountered
fun test(s: ~i32): ~i32 {
    TakeUntil(fun(x): x > 100) (s)
}
```

## Example 2

`Fold` is a task which aggregates a stream into a single value.

```text
task Fold[A,T](init: A, f: fun(A, T): A): ~T -> A {
    var agg: A = init;
    on {
        event => agg = f(agg, event),
        done => exit agg
    }
}

# Calculate the average of a stream
fun avg(s: ~i32): i32 {
    val sum   = Fold(0, fun(a,x): a + x) (s);
    val count = Fold(0, fun(a,_): a + 1) (s);
    sum / count
}
```


## Example 3

`MergeFold` is a task which aggregates two streams into a single value.

```text
task MergeFold[A,T](init: A, f: fun(A, T): A): (A(~T), B(~T)) -> A {
    var agg: A = init;
    on {
        A(event) or B(event) => agg = f(agg, event),
        done => exit agg
    }
}
```
