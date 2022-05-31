# ANCHOR: unit
@{rust: "unit"}
extern type unit;
# ANCHOR_END: unit

@{rust: "never"}
extern type never;

@{rust: "i8", mlir: "si8"}
extern type i8;

@{rust: "i16", mlir: "si16"}
extern type i16;

# ------------------------------------------------------

@{rust: "i32", mlir: "si32"}
extern type i32;

@{rust: "i32::add", mlir: "add_i32"}
extern def +(i32, i32): i32;

@{rust: "i32::sub", mlir: "sub_i32"}
extern def -(i32, i32): i32;

@{rust: "i32::mul", mlir: "mul_i32"}
extern def *(i32, i32): i32;

@{rust: "i32::div", mlir: "div_i32"}
extern def /(i32, i32): i32;

@{rust: "i32::pow", mlir: "pow_i32"}
extern def **(i32, i32): i32;

@{rust: "i32::rem", mlir: "rem_i32"}
extern def %(i32, i32): i32;

@{rust: "i32::eq", mlir: "eq_i32"}
extern def ==[T](T, T): bool;

@{rust: "i32::ge", mlir: "geq_i32"}
extern def >=(i32, i32): bool;

@{rust: "i32::le", mlir: "leq_i32"}
extern def <=(i32, i32): bool;

@{rust: "i32::gt", mlir: "gt_i32"}
extern def >(i32, i32): bool;

@{rust: "i32::lt", mlir: "lt_i32"}
extern def <(i32, i32): bool;

@{rust: "i32::or", mlir: "or_i32"}
extern def bor(i32, i32): bool;

@{rust: "i32::xor", mlir: "xor_i32"}
extern def bxor(i32, i32): bool;

@{rust: "i32::and", mlir: "and_i32"}
extern def band(i32, i32): bool;

@{rust: "i32::neg", mlir: "neg_i32"}
extern def neg::[i32](i32): i32;

@{rust: "Str::from_i32"}
extern def i32_to_string(i32): String;

# ------------------------------------------------------

@{rust: "i64", mlir: "si64"}
extern type i64;

@{rust: "i128", mlir: "si128"}
extern type i128;

@{rust: "u8", mlir: "ui8"}
extern type u8;

@{rust: "u16", mlir: "ui16"}
extern type u16;

@{rust: "u32", mlir: "ui32"}
extern type u32;

# @{rust: "u32::add", mlir: "add_u32"}
# extern def +::[u32](u32, u32): u32;
#
# @{rust: "u32::sub", mlir: "sub_u32"}
# extern def -::[u32](u32, u32): u32;
#
# @{rust: "u32::mul", mlir: "mul_u32"}
# extern def *::[u32](u32, u32): u32;
#
# @{rust: "u32::div", mlir: "div_u32"}
# extern def /::[u32](u32, u32): u32;
#
# @{rust: "u32::pow", mlir: "pow_u32"}
# extern def **::[u32](u32, u32): u32;
#
# @{rust: "u32::mod", mlir: "rem_u32"}
# extern def %::[u32](u32, u32): u32;
#
# @{rust: "u32::eq", mlir: "eq_u32"}
# extern def ==::[u32](u32, u32): bool;
#
# @{rust: "u32::ge", mlir: "geq_u32"}
# extern def >=::[u32](u32, u32): bool;
#
# @{rust: "u32::le", mlir: "leq_u32"}
# extern def <=::[u32](u32, u32): bool;
#
# @{rust: "u32::gt", mlir: "gt_u32"}
# extern def >::[u32](u32, u32): bool;
#
# @{rust: "u32::lt", mlir: "lt_u32"}
# extern def <::[u32](u32, u32): bool;
#
# @{rust: "u32::neg", mlir: "neg_u32"}
# extern def negu32(u32): u32;

# ------------------------------------------------------

@{rust: "u64", mlir: "u64"}
extern type u64;

@{rust: "u128", mlir: "u128"}
extern type u128;

# ------------------------------------------------------

@{rust: "f32", mlir: "f32"}
extern type f32;

# @{rust: "f32::add", mlir: "add_f32"}
# extern def +::[f32](f32, f32): f32;
#
# @{rust: "f32::sub", mlir: "sub_f32"}
# extern def -::[f32](f32, f32): f32;
#
# @{rust: "f32::sub", mlir: "mul_f32"}
# extern def *::[f32](f32, f32): f32;
#
# @{rust: "f32::sub", mlir: "div_f32"}
# extern def /::[f32](f32, f32): f32;
#
# @{rust: "f32::pow", mlir: "pow_f32"}
# extern def **::[f32](f32, f32): f32;
#
# @{rust: "f32::mod", mlir: "rem_f32"}
# extern def %::[f32](f32, f32): f32;
#
# @{rust: "f32::eq", mlir: "eq_f32"}
# extern def ==::[f32](f32, f32): bool;
#
# @{rust: "f32::le", mlir: "geq_f32"}
# extern def >=::[f32](f32, f32): bool;
#
# @{rust: "f32::ge", mlir: "leq_f32"}
# extern def <=::[f32](f32, f32): bool;
#
# @{rust: "f32::lt", mlir: "gt_f32"}
# extern def >::[f32](f32, f32): bool;
#
# @{rust: "f32::gt", mlir: "lt_f32"}
# extern def <::[f32](f32, f32): bool;
#
# @{rust: "f32::neg", mlir: "neg_f32"}
# extern def neg::[f32](f32): f32;

@{rust: "Str::from_f32"}
extern def f32_to_string(i32): String;

# ------------------------------------------------------

@{rust: "f64", mlir: "f64"}
extern type f64;

# ------------------------------------------------------

@{rust: "bool", mlir: "i1"}
extern type bool;

@{rust: "and", mlir: "and_i1"}
extern def and(bool, bool): bool;

@{rust: "or", mlir: "or_i1"}
extern def or(bool, bool): bool;

@{rust: "xor", mlir: "xor_i1"}
extern def xor(bool, bool): bool;

# @{rust: "bool::eq", mlir: "eq_i1"}
# extern def ==::[bool](bool, bool): bool;

def not(c) = if c { false } else { true }

# ------------------------------------------------------

@{rust: "char"}
extern type char;

# ------------------------------------------------------

# ANCHOR: string
@{rust: "Str"}
extern type String;

@{rust: "&'static str"}
extern type str;

@{rust: "Str::new"}
extern def new_str(): String;

@{rust: "Str::from_str"}
extern def from_str(str): String;

@{rust: "Str::concat"}
extern def concat(String, String): String;

@{rust: "Str::with_capacity"}
extern def str_with_capacity(u32): String;

@{rust: "Str::push_char"}
extern def push_char(String, char): unit;

@{rust: "Str::remove_char"}
extern def remove_char(String, u32): char;

@{rust: "Str::insert_char"}
extern def insert_char(String, u32, char): unit;

@{rust: "Str::is_empty"}
extern def is_empty_str(String): bool;

@{rust: "Str::split_off"}
extern def split_off(String, u32): String;

@{rust: "Str::clear"}
extern def clear_str(String);

@{rust: "Str::eq"}
extern def str_eq(String, String): bool;

# ANCHOR_END: string

# ------------------------------------------------------

@{rust: "bool_assert"}
extern def assert(bool);

@{rust: "Str::print"}
extern def print(String);

@{rust: "Str::panic"}
extern def panic(String);

# ------------------------------------------------------

# ANCHOR: array
@{rust: "Vector"}
extern type Array[T];
# ANCHOR_END: array

@{rust: "Vector::new"}
extern def array[T](): Array[T];

@{rust: "Vector::push"}
extern def push_back[T](Array[T], T);

@{rust: "Vector::pop"}
extern def pop_back[T](Array[T]);

@{rust: "Vector::remove"}
extern def remove[T](Array[T], i32): T;

@{rust: "Vector::select"}
extern def get[T](Array[T], i32): T;

@{rust: "Vector::insert"}
extern def insert[T](Array[T], i32, T);

@{rust: "Vector::replace"}
extern def replace[T](Array[T], i32, T);

@{rust: "Vector::is_empty"}
extern def is_empty[T](Array[T]): bool;

@{rust: "Vector::len"}
extern def len[T](Array[T]): i32;

@{rust: "Vector::extend"}
extern def extend[T](Array[T], Array[T]);

@{rust: "Vector::clear"}
extern def clear[T](Array[T]);

@{rust: "Vector::capacity"}
extern def capacity[T](Array[T]): i32;

# ------------------------------------------------------

@{rust: "Cell"}
extern type Cell[T];

@{rust: "Cell::new"}
extern def new_cell[T](T): Cell[T];

@{rust: "Cell::set"}
extern def set_cell[T](Cell[T], T);

@{rust: "Cell::get"}
extern def get_cell[T](Cell[T]): T;

# ------------------------------------------------------

@{rust: "Iter"}
extern type Iter[T];

@{rust: "Iter::next"}
extern def next[T](Iter[T], T): T;

# ------------------------------------------------------

@{rust: "Range"}
extern type Range[T];

@{rust: "Range::new"}
extern def rexc[T](T, T): Range[T];

@{rust: "Range::leq"}
extern def leq_rexc[T](Range[T], T): bool;

@{rust: "Range::geq"}
extern def geq_rexc[T](Range[T], T): bool;

@{rust: "Range::lt"}
extern def lt_rexc[T](Range[T], T): bool;

@{rust: "Range::gt"}
extern def gt_rexc[T](Range[T], T): bool;

# ------------------------------------------------------

@{rust: "PullChan"}
extern type PullChan[T];

@{rust: "PushChan"}
extern type PushChan[T];

@{rust: "PullChan::pull"}
extern async def pull[T](PullChan[T]): T;

@{rust: "PushChan::push"}
extern async def push[T](PushChan[T], T);

@{rust: "channel"}
extern def chan[T](): #(PushChan[T], PullChan[T]);
