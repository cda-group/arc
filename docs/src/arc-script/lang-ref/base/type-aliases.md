# Type Aliases

A **type alias** is a purely cosmetic alias of a type.

```text
Item ::=
  | 'type' Name '=' Type ';'  # Type alias
  | ...
```

## Example

The following code defines type aliases for representing lines on a two-dimensional plane, and a function for calculating the length of a line.

```text
type Num = i32;
type Point = {x: Num, y: Num}
type Line = {start: Point, end: Point}

fun length(line: Line): i32 {
    val a = line.start.x - line.end.x;
    val b = line.start.y - line.end.y;
    sqrt(a ** 2 + b ** 2)
}

fun test() {
    val p0 = { x: 0, y: 1 };
    val p1 = { x: 5, y: 9 };
    val line = { start: p0, end: p1 };
    print(length(line));
}
```
