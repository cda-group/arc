# Cells

A **cell** is an abstract data type which can be used to store a single value.

```text
extern type Cell[T](v: T) {
    fun set(x: T);
    fun get(): T;
}
```

Cells are useful for representing mutable state inside tasks. For this reason, all mutable variables inside tasks are desugared into cells.

## Example

`Scan` is a task which is a combination of `Map` and `Reduce`. It maintains an aggregate and outputs it for each aggregated element.

```text
task Scan[A, T](init: A, f: fun(A, T): A): ~T -> ~A {
    var agg = init;
    on event => {
        agg = f(agg, event);
        emit agg
    }
}
```

Implicitly, the above desugars to:

```text
task Scan[A, T](init: A, f: fun(A, T): A): ~T -> ~A {
    val agg = Cell(init);
    on event => {
        agg.set(f(agg.get(), event));
        emit agg.get()
    }
}
```
