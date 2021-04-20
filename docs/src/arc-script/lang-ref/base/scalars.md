# Scalars

A **scalar** is a primitive type which has a direct correspondence to **literals**.

```text
Scalar ::=
  | 'i8'  | 'i16'  | 'i32' | 'i64'  # Machine integers
  | 'u8'  | 'u16'  | 'u32' | 'u64'  # Machine unsigned integers
  | 'f16' | 'bf16' | 'f32' | 'f64'  # Machine floating points
  | 'bignum'                        # Arbitrary sized integer
  | 'bool'                          # Booleans
  | 'unit'                          # Unit
  | 'char' | 'str'                  # UTF8-encoded chars and strings
  | ..
```
