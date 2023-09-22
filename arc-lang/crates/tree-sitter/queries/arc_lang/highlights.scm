;; Literals

(constant_string (string_quote) @string)
(constant_string (string_content) @string)
(constant_int) @number
(constant_bool) @boolean
(constant_float) @float
(constant_char) @character
(string_interpolation (block) @string.special)
(string_interpolation (name) @string.special)
(line_comment) @comment

(stmt_code "---" @comment)

;; Identifiers

(expr_name (name) @variable)
(type_name (name) @type)

(pattern_name (name) @variable)
(stmt_def name: (name) @function)
(stmt_type name: (name) @type)
(stmt_enum name: (name) @type)

; (expr_call function: (expr_name (name) @function))
; (expr_method_call name: ((name) @function))

((type_name) @type.builtin
 (#any-of? @type.builtin
    "i8" "i16" "i32" "i64" "i128"
    "f32" "f64"
    "u8" "u16" "u32" "u64" "u128"
    "bool" "char" "String"
    "Option" "Result"
    "Vec" "Dict" "Stream"
    "Time" "Duration" "SocketAddress" "File" "Path" "Url"
))

;; Keywords

[ "if" "else" "match" "finally" "try" "catch" "throw" ] @conditional

[ "loop" "while" "for" "break" "continue"
  "from" "in" "as" "desc" "group" "into" "join" "on" "of" "select" "compute" "union" "over" "roll" "order" "where" "with"
  ] @repeat

[
  "def" "do" "fun" "return"
  "type" "val" "var" "enum"
  "rust" "python"
] @keyword

;; Punctuation

[ ";" ":" "::" "," "." "@" "{" "}" "[" "]" "(" ")" "=" "=>" "_" ] @punctuation

[ "not" "and" "or" "+" "-" "*" "/" "==" "!=" "<" ">" "<=" ">=" "+=" "-=" "*=" "/=" ".." "..=" "&" "++" ] @operator
