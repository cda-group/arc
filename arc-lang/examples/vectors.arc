# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
val a: Array[i32] = [1, 2, 3];

val b: Array[i32] = array();
b.push(1);
b.push(2);
b.push(3);

assert(a[0u32] == b[0u32]);
assert(a[1u32] == b[1u32]);
assert(a[2u32] == b[2u32]);

a[0u32] = 2;

assert(a[0u32] == 2);
assert(a.get(0u32) == 2);

# for x in a {
#     assert(x != 0);
# }

assert(a.len() ==u32 3u32);
assert(a.capacity() ==u32 3u32);
assert(not a.is_empty());

a.clear();
assert(a.is_empty());

b.pop();
assert(b.len() ==u32 2u32);

val c = [1];

c.extend([2, 3]);
assert(c[0u32] == 1);
assert(c[1u32] == 2);
assert(c[2u32] == 3);

c.remove(0u32);
assert(c[0u32] == 2);

c.insert(0u32, 1);
assert(c[0u32] == 2);

# ANCHOR_END: example
}

