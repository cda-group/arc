# ANCHOR: unit
@{intrinsic: "none"}
extern type unit;
# ANCHOR_END: unit

@{intrinsic: "si8"}
extern type i8;

@{intrinsic: "si16"}
extern type i16;

# ------------------------------------------------------

@{intrinsic: "si32"}
extern type i32;

@{intrinsic: "add_i32"}
extern def +(i32, i32): i32;

@{intrinsic: "sub_i32"}
extern def -(i32, i32): i32;

@{intrinsic: "mul_i32"}
extern def *(i32, i32): i32;

@{intrinsic: "div_i32"}
extern def /(i32, i32): i32;

@{intrinsic: "pow_i32"}
extern def **(i32, i32): i32;

@{intrinsic: "rem_i32"}
extern def %(i32, i32): i32;

@{intrinsic: "eq_i32"}
extern def ==(i32, i32): bool;

@{intrinsic: "geq_i32"}
extern def >=(i32, i32): bool;

@{intrinsic: "leq_i32"}
extern def <=(i32, i32): bool;

@{intrinsic: "gt_i32"}
extern def >(i32, i32): bool;

@{intrinsic: "lt_i32"}
extern def <(i32, i32): bool;

@{intrinsic: "or_i32"}
extern def bor(i32, i32): bool;

@{intrinsic: "xor_i32"}
extern def bxor(i32, i32): bool;

@{intrinsic: "and_i32"}
extern def band(i32, i32): bool;

@{intrinsic: "neg_i32"}
extern def neg(i32): i32;

extern def i32_to_string(i32): String;

# ------------------------------------------------------

@{intrinsic: "si64"}
extern type i64;

@{intrinsic: "u64"}
extern type u8;

@{intrinsic: "u128"}
extern type u16;

@{intrinsic: "u64"}
extern type u32;

@{intrinsic: "add_u32"}
extern def +u32(u32, u32): u32;

@{intrinsic: "sub_u32"}
extern def -u32(u32, u32): u32;

@{intrinsic: "mul_u32"}
extern def *u32(u32, u32): u32;

@{intrinsic: "div_u32"}
extern def /u32(u32, u32): u32;

@{intrinsic: "pow_u32"}
extern def **u32(u32, u32): u32;

@{intrinsic: "rem_u32"}
extern def %u32(u32, u32): u32;

@{intrinsic: "eq_u32"}
extern def ==u32(u32, u32): bool;

@{intrinsic: "geq_u32"}
extern def >=u32(u32, u32): bool;

@{intrinsic: "leq_u32"}
extern def <=u32(u32, u32): bool;

@{intrinsic: "gt_u32"}
extern def >u32(u32, u32): bool;

@{intrinsic: "lt_u32"}
extern def <u32(u32, u32): bool;

@{intrinsic: "neg_u32"}
extern def negu32(u32): u32;

# ------------------------------------------------------

@{intrinsic: "u64"}
extern type u64;

@{intrinsic: "u128"}
extern type u128;

# ------------------------------------------------------

@{intrinsic: "f32"}
extern type f32;

@{intrinsic: "add_f32"}
extern def +f32(f32, f32): f32;

@{intrinsic: "sub_f32"}
extern def -f32(f32, f32): f32;

@{intrinsic: "mul_f32"}
extern def *f32(f32, f32): f32;

@{intrinsic: "div_f32"}
extern def /f32(f32, f32): f32;

@{intrinsic: "pow_f32"}
extern def **f32(f32, f32): f32;

@{intrinsic: "rem_f32"}
extern def %f32(f32, f32): f32;

@{intrinsic: "eq_f32"}
extern def ==f32(f32, f32): bool;

@{intrinsic: "geq_f32"}
extern def >=f32(f32, f32): bool;

@{intrinsic: "leq_f32"}
extern def <=f32(f32, f32): bool;

@{intrinsic: "gt_f32"}
extern def >f32(f32, f32): bool;

@{intrinsic: "lt_f32"}
extern def <f32(f32, f32): bool;

@{intrinsic: "neg_f32"}
extern def negf32(f32): f32;

extern def f32_to_string(i32): String;

# ------------------------------------------------------

@{intrinsic: "f64"}
extern type f64;

# ------------------------------------------------------

@{intrinsic: "i1"}
extern type bool;

@{intrinsic: "and_i1"}
extern def and(bool, bool): bool;

@{intrinsic: "or_i1"}
extern def or(bool, bool): bool;

@{intrinsic: "xor_i1"}
extern def xor(bool, bool): bool;

@{intrinsic: "eq_i1"}
extern def ==bool(bool, bool): bool;

def not(c) = if c ==bool true { false } else { true }

# ------------------------------------------------------

extern type char;

# ------------------------------------------------------

# ANCHOR: string
extern type String;

extern def concat(String, String): String;
# ANCHOR_END: string

# ------------------------------------------------------

extern def assert(bool);

extern def print(String);

extern def panic(String);

# ------------------------------------------------------

# ANCHOR: option
enum Option[T] {
    Some(T),
    None
}
# ANCHOR_END: option

# ------------------------------------------------------

# ANCHOR: array
extern type Array[T];
# ANCHOR_END: array

extern def array[T](): Array[T];
extern def push[T](Array[T], T);
extern def pop[T](Array[T]);
extern def select[T](Array[T], i32): T;
extern def len[T](Array[T]): i32;
extern def extend[T](Array[T], Array[T]);
extern def contains[T](T, Array[T]): bool;

# ------------------------------------------------------

extern type Cell[T];

extern def cell[T](T): Cell[T];
extern def update[T](Cell[T], T);
extern def read[T](Cell[T]): T;

# ------------------------------------------------------

extern type Iter[T];

extern def next[T](Iter[T], T): T;

# ------------------------------------------------------

extern type Range[T];

extern def new_range[T](T, T): Range[T];
extern def leq_range[T](Range[T], T): bool;
extern def geq_range[T](Range[T], T): bool;
extern def lt_range[T](Range[T], T): bool;
extern def gt_range[T](Range[T], T): bool;

# ------------------------------------------------------

extern type Stream[T];

extern def map[A,B](Stream[A], fun(A):B): Stream[B];
extern def key_by[K,V](Stream[V], fun(V):K): KStream[K,V];

# ------------------------------------------------------

extern type KStream[K,V];

extern def fold[K,V](KStream[K,V], fun(V,V):V): Stream[V];
