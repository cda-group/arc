# ANCHOR: unit
@{mlir: "none"}
extern type unit;
# ANCHOR_END: unit

@{mlir: "si8"}
extern type i8;

@{mlir: "si16"}
extern type i16;

# ------------------------------------------------------

@{mlir: "si32"}
extern type i32;

@{mlir: "add_i32"}
extern def +(i32, i32): i32;

@{mlir: "sub_i32"}
extern def -(i32, i32): i32;

@{mlir: "mul_i32"}
extern def *(i32, i32): i32;

@{mlir: "div_i32"}
extern def /(i32, i32): i32;

@{mlir: "pow_i32"}
extern def **(i32, i32): i32;

@{mlir: "rem_i32"}
extern def %(i32, i32): i32;

@{mlir: "eq_i32"}
extern def ==(i32, i32): bool;

@{mlir: "geq_i32"}
extern def >=(i32, i32): bool;

@{mlir: "leq_i32"}
extern def <=(i32, i32): bool;

@{mlir: "gt_i32"}
extern def >(i32, i32): bool;

@{mlir: "lt_i32"}
extern def <(i32, i32): bool;

@{mlir: "or_i32"}
extern def bor(i32, i32): bool;

@{mlir: "xor_i32"}
extern def bxor(i32, i32): bool;

@{mlir: "and_i32"}
extern def band(i32, i32): bool;

@{mlir: "neg_i32"}
extern def neg(i32): i32;

@{rust: "String_from_i32"}
extern def i32_to_string(i32): String;

# ------------------------------------------------------

@{mlir: "si64"}
extern type i64;

@{mlir: "si128"}
extern type i128;

@{mlir: "ui8"}
extern type u8;

@{mlir: "ui16"}
extern type u16;

@{mlir: "ui32"}
extern type u32;

@{mlir: "add_u32"}
extern def +u32(u32, u32): u32;

@{mlir: "sub_u32"}
extern def -u32(u32, u32): u32;

@{mlir: "mul_u32"}
extern def *u32(u32, u32): u32;

@{mlir: "div_u32"}
extern def /u32(u32, u32): u32;

@{mlir: "pow_u32"}
extern def **u32(u32, u32): u32;

@{mlir: "rem_u32"}
extern def %u32(u32, u32): u32;

@{mlir: "eq_u32"}
extern def ==u32(u32, u32): bool;

@{mlir: "geq_u32"}
extern def >=u32(u32, u32): bool;

@{mlir: "leq_u32"}
extern def <=u32(u32, u32): bool;

@{mlir: "gt_u32"}
extern def >u32(u32, u32): bool;

@{mlir: "lt_u32"}
extern def <u32(u32, u32): bool;

@{mlir: "neg_u32"}
extern def negu32(u32): u32;

# ------------------------------------------------------

@{mlir: "u64"}
extern type u64;

@{mlir: "u128"}
extern type u128;

# ------------------------------------------------------

@{mlir: "f32"}
extern type f32;

@{mlir: "add_f32"}
extern def +f32(f32, f32): f32;

@{mlir: "sub_f32"}
extern def -f32(f32, f32): f32;

@{mlir: "mul_f32"}
extern def *f32(f32, f32): f32;

@{mlir: "div_f32"}
extern def /f32(f32, f32): f32;

@{mlir: "pow_f32"}
extern def **f32(f32, f32): f32;

@{mlir: "rem_f32"}
extern def %f32(f32, f32): f32;

@{mlir: "eq_f32"}
extern def ==f32(f32, f32): bool;

@{mlir: "geq_f32"}
extern def >=f32(f32, f32): bool;

@{mlir: "leq_f32"}
extern def <=f32(f32, f32): bool;

@{mlir: "gt_f32"}
extern def >f32(f32, f32): bool;

@{mlir: "lt_f32"}
extern def <f32(f32, f32): bool;

@{mlir: "neg_f32"}
extern def negf32(f32): f32;

@{mlir: "String_from_f32"}
extern def f32_to_string(i32): String;

# ------------------------------------------------------

@{mlir: "f64"}
extern type f64;

# ------------------------------------------------------

@{mlir: "i1"}
extern type bool;

@{mlir: "and_i1"}
extern def and(bool, bool): bool;

@{mlir: "or_i1"}
extern def or(bool, bool): bool;

@{mlir: "xor_i1"}
extern def xor(bool, bool): bool;

@{mlir: "eq_i1"}
extern def ==bool(bool, bool): bool;

def not(c) = if c ==bool true { false } else { true }

# ------------------------------------------------------

@{rust: "char"}
extern type char;

# ------------------------------------------------------

# ANCHOR: string
@{rust: "String"}
extern type String;

@{rust: "&'static str"}
extern type str;

@{rust: "String_new"}
extern def new_str(): String;

@{rust: "String_from_str"}
extern def from_str(str): String;

@{rust: "String_concat"}
extern def concat(String, String): String;

@{rust: "String_with_capacity"}
extern def str_with_capacity(u32): String;

@{rust: "String_push_char"}
extern def push_char(String, char): unit;

@{rust: "String_remove_char"}
extern def remove_char(String, u32): char;

@{rust: "String_insert_char"}
extern def insert_char(String, u32, char): unit;

@{rust: "String_is_empty"}
extern def is_empty_str(String): bool;

@{rust: "String_split_off"}
extern def split_off(String, u32): String;

@{rust: "String_clear"}
extern def clear_str(String);

@{rust: "String_eq"}
extern def str_eq(String, String): bool;

# ANCHOR_END: string

# ------------------------------------------------------

@{rust: "assert"}
extern def assert(bool);

@{rust: "print"}
extern def print(String);

@{rust: "panic"}
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
@{rust: "Vec"}
extern type Array[T];
# ANCHOR_END: array

@{rust: "Vec_new"}
extern def array[T](): Array[T];

@{rust: "Vec_push"}
extern def push[T](Array[T], T);

@{rust: "Vec_pop"}
extern def pop[T](Array[T]);

@{rust: "Vec_remove"}
extern def remove[T](Array[T], u32): T;

@{rust: "Vec_select"}
extern def get[T](Array[T], u32): T;

@{rust: "Vec_insert"}
extern def insert[T](Array[T], u32, T);

@{rust: "Vec_replace"}
extern def replace[T](Array[T], u32, T);

@{rust: "Vec_is_empty"}
extern def is_empty[T](Array[T]): bool;

@{rust: "Vec_len"}
extern def len[T](Array[T]): u32;

@{rust: "Vec_extend"}
extern def extend[T](Array[T], Array[T]);

@{rust: "Vec_clear"}
extern def clear[T](Array[T]);

@{rust: "Vec_capacity"}
extern def capacity[T](Array[T]): u32;

# ------------------------------------------------------

@{rust: "Cell"}
extern type Cell[T];

@{rust: "Cell_new"}
extern def cell[T](T): Cell[T];

@{rust: "Cell_set"}
extern def set_cell[T](Cell[T], T);

@{rust: "Cell_get"}
extern def get_cell[T](Cell[T]): T;

# ------------------------------------------------------

@{rust: "Iter"}
extern type Iter[T];

@{rust: "Iter_next"}
extern def next[T](Iter[T], T): T;

# ------------------------------------------------------

@{rust: "Range"}
extern type Range[T];

@{rust: "Range_new"}
extern def new_range[T](T, T): Range[T];

@{rust: "Range_leq"}
extern def leq_range[T](Range[T], T): bool;

@{rust: "Range_geq"}
extern def geq_range[T](Range[T], T): bool;

@{rust: "Range_lt"}
extern def lt_range[T](Range[T], T): bool;

@{rust: "Range_gt"}
extern def gt_range[T](Range[T], T): bool;

# ------------------------------------------------------

@{rust: "Stream"}
extern type Stream[T];

@{rust: "Stream_map"}
extern def map[A,B](Stream[A], fun(A):B): Stream[B];
