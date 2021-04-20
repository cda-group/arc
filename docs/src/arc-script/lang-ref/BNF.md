In the following sections, we will explain each concept by formally defining its syntax and by informally discussing its semantics. Syntax will be explained using a Regex-based variation of the BNF grammar where:

* `+` and `*` denote repetition.
* `?` is for optional rules.
* `(` `)` indicates grouping.
* `|` is for alternation.
* `[` `]` for is character-alternation (e.g., `[abc]`).
* `-` is for ranges (e.g., `a-zA-Z`).
* `.` is for matching any character.
* `\` is for escaping characters.
* Rules are written as uppercase (e.g., `Rule`).
* Symbols are enclosed with single-quotes (e.g., `'symbol'`).
