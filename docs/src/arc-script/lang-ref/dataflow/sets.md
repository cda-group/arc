# Sets

A **set** is an abstract data type which can be used to store collections of unique values.

```text
extern type Set[T]() {
    fun add(elem: T);
    fun delete(elem: T);
    fun contains(elem: T): bool;
}
```

Sets have the following syntactic sugar:

```text
Expr ::=
  | Expr 'in' Expr        # Check if set contains element
  | Expr 'not' 'in' Expr  # Check if does not contain element
  | ..
```

## Example

`Deduplicate` is a task which filters out unique numbers from a stream of integers.

```text
task Deduplicate(): ~i32 -> ~i32 {
    val unique: Set[i32] = Set();
    on event => if event not in unique {
        unique.add(event);
        emit event
    }
}
```
