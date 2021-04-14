# Appenders

An *appender* is an append-only vector which can be folded. By folding, the appender's elements are consumed, and the appender is cleared.

```text
extern type Appender[T]() {
    fun push(key: T);
    fun fold[A](init: A, fun(A, T): A): A;
    fun clear();
}
```

## Example

`LazyFold` is a task which starts folding as soon as a predicate is satisfied:

```text
task LazyFold[A,T](pred: fun(T) -> bool, init: A, binop: fun(A, T): A): ~T -> ~A {
    val app: Appender[T] = Appender();
    on event => {
        app.push(event);
        if pred(event) {
            emit app.fold(init, binop)
        }
    }
}
```
