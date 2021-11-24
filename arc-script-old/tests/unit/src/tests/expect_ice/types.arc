type Num = i32;

type Point = { x: Num, y: Num };

enum Direction { East, West, South, North };

type Arrow = { point: Point, dir: Direction };

enum Thing {
    A(i32),
    B(i32),
    C(i32),
    D(i32),
}

