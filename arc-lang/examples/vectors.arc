# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
val a: [i32] = [1, 2, 3];

val b: Array[i32] = array();
b.push_back(1);
b.push_back(2);
b.push_back(3);

assert(a[0] == b[0]);
assert(a[1] == b[1]);
assert(a[2] == b[2]);

a[0] = 2;

assert(a[0] == 2);
assert(a.get(0) == 2);

# for x in a {
#     assert(x != 0);
# }

assert(a.len() == 3);
assert(a.capacity() == 3);
assert(not a.is_empty());

a.clear();
assert(a.is_empty());

b.pop_back();
assert(b.len() == 2);

val c = [1];

c.extend([2, 3]);
assert(c[0] == 1);
assert(c[1] == 2);
assert(c[2] == 3);

c.remove(0);
assert(c[0] == 2);

c.insert(0, 1);
assert(c[0] == 2);

# ANCHOR_END: example
}

