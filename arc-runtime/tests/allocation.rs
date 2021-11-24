use comet::api::Collectable;
use comet::api::Finalize;
use comet::api::Gc;
use comet::api::Trace;
use comet::api::Visitor;
use comet::gc_base::AllocationSpace;
use comet::immix::instantiate_immix;
use comet::immix::Immix;
use comet::immix::ImmixOptions;

pub enum List<T: Collectable> {
    Nil,
    Cons(T, Gc<List<T>, Immix>),
}

unsafe impl<T: Collectable> Trace for List<T> {
    fn trace(&mut self, vis: &mut dyn Visitor) {
        if let Self::Cons(data, next) = self {
            data.trace(vis);
            next.trace(vis);
        }
    }
}

unsafe impl<T: Collectable> Finalize for List<T> {}

impl<T: Collectable> Collectable for List<T> {}

#[test]
fn main() {
    let opts = ImmixOptions::default();
    let mut mutator = instantiate_immix(opts);

    let mut l: Gc<List<i32>, Immix> = mutator.allocate(List::Nil, AllocationSpace::New);

    for i in 0..100 {
        l = mutator.allocate(List::Cons(i, l), AllocationSpace::New);
    }

    for i in 0..100 {
        match *l {
            List::Nil => break,
            List::Cons(data, tail) => {
                assert_eq!(data, 99 - i);
                l = tail;
            }
        }
    }
}
