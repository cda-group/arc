# Enums

An **enum** is a nominal type which is a disjoint set (tagged union) of values. Each item of the disjoint set is referred to as a **variant**.

```text
Item ::=
  | 'enum' Name '{' (Name '(' Type ')' ',')* '}' # Enumerated type (Sum-type)
  | ...

Expr ::=
  | Path '(' Expr ')'  # Enum-variant consruction
  | ...
```

## Examples

In the following code, we create a `Shape`-enum type which can hold `Rectangle`-shapes and `Circle`-shapes. Then, we define function `area` is defined for calculating the area of a `Shape`. Note how pattern matching is required to extract values from enums.

```text
enum Shape {
    Rectangle({ width: f32, height: f32 }),
    Circle({ radius: f32 }),
}

fun area(shape: Shape): f32 {
    match shape {
        Shape::Rectangle({width, height}) => width * height,
        Shape::Circle({radius}) => PI * radius ** 2,
    }
}

fun test() {
    val rectangle = { width: 5.0, height: 3.0 };
    val shape = Shape::Rectangle(rectangle);
    print(area(shape));
}
```
