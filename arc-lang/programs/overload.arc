
def id[Self](a: Self): Self;

def id(a) for [i32] { a }

def id(a) for [f32] { a }

def id[T](a) for [T] { a }

------------

def id[T](a: Self, b: T): Self;

def id[T](a, b) for i32 { a }

------------

trait Id[T] {
    def id(a: Self, b: T): Self;
}

impl Id for i32 {
    fn id[T](a, b: T) -> T {
        a
    }
}
