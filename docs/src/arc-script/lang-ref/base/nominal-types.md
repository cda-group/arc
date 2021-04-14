# Nominal types

A **nominal type** is a type which is unified by-name as opposed to by-structure. Nominals resemble enums with a single anonymous variant.

```text
Item ::=
  | 'enum' Name '(' Type ')' ';' # Nominal type
  | ...

Expr ::=
  | Path '(' Expr ')'            # Nominal construction
  | ...
```

## Example

The following code shows how to create nominal types `Human` and `Alien`.

```text
enum Human({age:i32, name:str});
enum Alien({age:i32, name:str});

fun print(Human(h): Human) {
    print(h.name)
}

fun test() {
    let h = Human({age: 5, name: "bob"});
    let a = Alien({age: 5, name: "bob"});

    print(h); # OK
    print(a); # Error: found Alien, expected Human
}
```
