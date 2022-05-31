#![allow(clippy::blacklisted_name)]
#![allow(unused)]

use arc_runtime::prelude::Gc;
use arc_runtime::prelude::Heap;
use arc_runtime::prelude::Trace;

#[derive(Eq, PartialEq, Clone, Copy, Debug, Trace)]
enum List {
    Cons(i32, Gc<List>),
    Nil,
}

fn nil() -> List {
    List::Nil
}

fn cons(x: i32, list: Gc<List>) -> List {
    List::Cons(x, list)
}

#[test]
fn test() {
    let heap = Heap::new();
    let a = heap.allocate(cons(1, heap.allocate(cons(2, heap.allocate(nil())))));
    a.root(heap);
    let b = heap.allocate(cons(1, heap.allocate(cons(2, heap.allocate(nil())))));
    b.root(heap);
    let c = heap.allocate(cons(1, heap.allocate(cons(2, heap.allocate(nil())))));
    c.root(heap);

    // ...
    b.unroot(heap);
    heap.collect();
    assert_eq!(a, c);
    // assert_eq!(b, c); // <-- UNSAFE
}
