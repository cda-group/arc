# Keys

Events in streams are always partitioned by **key**. By default the **key** is implicit, but can be made explicit by annotating a function or task as `shuffle`. A task or function marked as `shuffle` is able to update its streams' keys.

```text
Expr ::=
  | Expr 'by' Expr
  | ..

Type ::=
  | Type 'by' Type
  | ..

Item ::=
  | 'shuffle' 'fun' Name '(' (Name ':' Type ',')+ ')' ':' Type '{' TaskItem* '}'
  | 'shuffle' 'task' Name '(' (Name ':' Type ',')+ ')' ':' Interface '->' Interface '{' TaskItem* '}'
  | ..
```

## Semantics

Keyed values, types, patterns, and expressions, are all desugared as follows:

```text
------------------------------------(DesugarBy)
Γ ⊢ t0 by t1  =  {val:t0, key:t1}
```

## Example 1

`KeyBy` changes the key of a stream from `K0` to `K1` through function `f`.

```text
shuffle task KeyBy[V, K0, K1](f: fun(V): K1) ~V by K0 -> ~V by K1 {
    on event by old_key => emit event by f(event)
}
```

An example of how to use `KeyBy`:

```text
shuffle fun test(s: ~i32 by i32): ~i32 by i32 {
    s |> KeyBy(fun(x): x % 50) # Re-shuffle into 50 key groups
      |> KeyBy(fun(x): x % 10) # Re-shuffle into 10 key groups
}
```

## Example 2

`Map` changes the value of a stream from `V0` to `V1` through function `f`, while leaving the key untouched.

```text
task Map[V0, V1](f: fun(V0): V1) ~V0 -> ~V1 {
    on event => emit f(event)
}
```

An example of how to use `KeyBy`:

```text
fun test(s: ~i32): ~i32 {
  s |> Map(fun(x): x + 1)
    |> Map(fun(x): x - 1)
}
```

Implicitly, the above will desugar into a generic keyed function:

```text
shuffle fun test[K](s: ~i32 by K): ~i32 by K {
    s |> Map(fun(x): x + 1)
      |> Map(fun(x): x - 1)
}
```
