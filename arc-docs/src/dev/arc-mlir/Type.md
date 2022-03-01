# Types

Unlike Arc-Lang, Arc-MLIR has no nominal types.

```grammar
Type ::=
  | "!arc.struct" "<" ([Name] ":" [Type])","* ">"
  | "!arc.enum" "<" ([Name] ":" [Type])","* ">"
  | "!arc.adt" "<" [String] ">"
  | "!arc.stream" "<" [Type] ">"
  | "f32" | "f64" | "i32" | "i64" | "si32" | "si64"
  | "()"
  | [[Deprecated]]

Deprecated ::= # Deprecated and unused types
  | "!arc.arcon.value" "<" [Type] ">"
  | "!arc.arcon.appender" "<" [Type] ">"
  | "!arc.arcon.map" "<" [Type]"," [Type] ">"

String ::= """[^"""]*"""
```
