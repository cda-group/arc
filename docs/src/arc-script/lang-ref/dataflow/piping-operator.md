# Piping Operator

The **piping operator** can be used to compose dataflow programs without creating intermediate variables.

```text
Expr ::=
  | Expr '|>' Expr
  | ..
```

## Semantics

The semantics of the piping operator is based on function application:

```text
 Γ ⊢ t0: A
 Γ ⊢ t1: A -> B
------------------------(Pipe)
 Γ ⊢ t0 |> t1  =  t1(t0)
```

## Examples

The following pipeline:

```text
fun pipeline(s0: ~i32): ~i32 {
    val s1 = Identity() (s0);
    val s2 = Identity() (s1);
    val s3 = Identity() (s2);
    s3
}
```

Could be rewritten as follows using the piping operator:

```text
fun pipeline(s0: ~i32): ~i32 {
    s0 |> Identity()
       |> Identity()
       |> Identity()
}
```
