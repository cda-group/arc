# Uses

A **use**-item imports a name into the current namespace and optionally aliases it.


```text
Item ::=
  | 'use' Path ('as' Name)?;  # Import (and optionally alias)
  | ...
```

## Examples

The following code creates a `Person`-type and an alias and use it as a `Human`.

```text
type Person = { name: str, age: i32 }

use Person as Human; # Creates an alias

fun print_name(human: Human): str {
    print(human.name)
}

fun test() {
    val person: Person = { name: str, age: i32 };
    print_name(person)
}
```
