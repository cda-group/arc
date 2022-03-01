# Operations

This section describes the operations of Arc-MLIR.

```grammar
Operation ::=
  | [[Arc]]
  | [[Std]]
  | [[Deprecated]]

Arc ::=
  | "arc.keep" "(" [Value] ")" ":" "(" [Type] ")" "->" "()"
  | "arc.tan" [Value] ":" [Type]
  | "arc.acos" [Value] ":" [Type]
  | "arc.asin" [Value] ":" [Type]
  | "arc.cosh" [Value] ":" [Type]
  | "arc.sinh" [Value] ":" [Type]
  | "arc.erf" [Value] ":" [Type]
  | "arc.make_struct" "(" [Value]","* ":" [Type]","* ")" ":" [Type]
  | "arc.make_enum" [Value] "(" [Value] ":" [Type] ")" ":" [Type]
  | "arc.if" "(" [Value] ")" "(" [Block]"," [Block] ")"
  | "arc.loop.break" "(" [Value]","+ ")" ":" "(" [Type]","+ ")" "->" "()"
  | "arc.adt_constant" [[String]] ":" [Type]
  | "arc.constant" [Literal] ":" [Type]
  | "arc.cmpi" [[Cmp]]"," [Value]"," [Value] ":" [Type]
  | "arc.receive" "(" [Value] ")" ":" "(" [Type] ")" "->" [Type]
  | "arc.select" [Value]"," [Value]"," [Value] ":" [Type]
  | "arc.send" "(" [Value]"," [Value] ")" "->" "()"
  | "arc.enum_access" [[String]] "in" "(" [Value] ":" [Type] ")" ":" [Type]
  | "arc.enum_check" "(" [Value] ":" [Type] ")" "is" [[String]] ":" [Type]
  | "arc.struct_access" "(" [Value] ")" "{" "field" "=" [[String]] "}" ":" "(" [Type] ")" "->" [Type]
  | "arc.addi" [Value]"," [Value] ":" [Type]
  | "arc.and" [Value]"," [Value] ":" [Type]
  | "arc.divi" [Value]"," [Value] ":" [Type]
  | "arc.or" [Value]"," [Value] ":" [Type]
  | "arc.muli" [Value]"," [Value] ":" [Type]
  | "arc.subi" [Value]"," [Value] ":" [Type]
  | "arc.remi" [Value]"," [Value] ":" [Type]
  | "arc.xor" [Value]"," [Value] ":" [Type]
  | "arc.panic" "()" ("msg" "=" [[String]])? : "()" "->" "()"

Std ::=
  | "call" [Path] "(" [Value]","+ ")" ":" "(" [Type]","+ ")" "->" [Type]
  | "call_indirect" [Value] "(" [Value]","+ ")" ":" "(" [Type]","+ ")" "->" [Type]
  | "return" "(" [Value] ")" ":" "(" [Type] ")" "->" "()"

Deprecated ::= # Deprecated and unused operations
  | "arc.emit" "(" [Value]"," [Value] ")" "->" "(" [Type]"," [Type] ")" "->" "()"
  | "arc.make_vector" "(" [Value]","* ")" ":" ([Type]","*) "->" [Type]
  | "arc.make_tuple" "(" [Value]","* ")" ":" "(" [Type]","* ")" "->" [Type]
  | "arc.make_tensor" "(" [Value]","* ")" ":" "(" [Type]","* ")"
  | "arc.index_tuple" "(" [Value] ")" "{" "index" "=" [[Int]] "}" ":" "(" [Type] ")" "->" [Type]
  | "arc.make_appender" "()" ":" "()" -> [Type]
  | "arc.merge" "(" [Value]"," [Value] ")" ":" "(" [Type]","* ")" "->" [Type]
  | "arc.result" "(" [Value] ")" ":" "(" [Type]","* ")" "->" [Type]
  | "arc.appender_push" "(" [Value]"," [Value] ")" ":" "(" [Type]"," [Type] ")" "->" "()"
  | "arc.appender_fold" "(" [Value]"," [Value] ")" ":" "(" [Type]"," [Type] ")" "->" [Type]
  | "arc.map_contains" "(" [Value]"," [Value] ")" ":" "(" [Type]"," [Type] ")" "->" [Type]
  | "arc.map_get" "(" [Value]"," [Value] ")" ":" "(" [Type]"," [Type] ")" "->" [Type]
  | "arc.map_insert" "(" [Value]"," [Value]"," [Value] ")" ":" "(" [Type]"," [Type]"," [Type] ")" "->" [Type]
  | "arc.map_remove" "(" [Value]"," [Value] ")" ":" "(" [Type]"," [Type] ")" "->" [Type]
  | "arc.value_write" "(" [Value]"," [Value] ")" ":" "(" [Type]"," [Type] ")" "->" [Type]
  | "arc.value_read" "(" [Value] ")" ":" "(" [Type] ")" "->" [Type]

Int ::= ["1"-"9"]["0"-"9"]*"."["0"-"9"]*

String ::= """[^"""]*"""

Literal ::=
  | Int
  | String

Cmp ::= "eq" | "ne" | "lt" | "le" | "gt" | "ge"
```

## Builtin functions

```arc-lang
{{#exec grep -h -o 'func [^{]*' ../arc-lang/stdlib/stdlib.mlir}}
```
