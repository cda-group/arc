def identity[A](x: A, y: A): A {
    identity(x, y)
}

def main() {
    val x: i32 = 5;
    identity::[i32](x, x);
}

task Identity[A](): ~A -> ~A {
    loop {
        on event => emit event;
    }
}
