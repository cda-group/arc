# Patterns

A **pattern** is a syntactic construct for deconstructing a **value** and binding its parts to variables.

```text
Pattern ::=
  | Name                                # Variable binding
  | Literal                             # Literal comparison
  | '{' (Name (':' Pattern)? ',')+ '}'  # Record deconstruction
  | '(' (Pattern ',')+ ')'              # Tuple deconstruction
  | Pattern? '..' ('='? Pattern)?       # Range deconstruction
  | Path '(' Pattern ')'                # Variant deconstruction
  | Pattern 'or' Pattern                # Alternation
```

## Example

Pattern matching on **tuples**:

```
val a = (1, 2, 3);
val (x0, x1, x2) = a;
```

Pattern matching on **records**, which also creates an alias `my_alias`:

```
val a = {x0:0, x1:2, x2:3};
val {x0, x1, x2: my_alias};
```
