# Dictionaries

A *dictionary* is an abstract data type for storing key-value pairs.

```text
extern type Dict[K,V]() {
    fun insert(key: K, val: V);
    fun get(key: K, default: V): V;
    fun contains(key: K): bool;
}
```

Dictionaries have the following syntactic sugar:

```text
Expr ::=
  | Expr 'in' Expr        # Check if dict contains key
  | Expr 'not' 'in' Expr  # Check if dict does not contain key
  | Expr '[' Expr ']'     # Get element
```

## Example

`WordCount` is a task which counts the occurrences of words in a stream of words:

```text
task WordCount(): ~str -> ~i32 {
    val count: Dict[str,i32] = Dict();
    on word => {
        count[word] = count.get(word, 0) + 1;
        emit count[word]
    }
}
```
